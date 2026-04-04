from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from whisper.whisper import Conv1d, LayerNorm, Linear, ModelDimensions, sinusoids


def _conv_output_length(lengths: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    return torch.div(lengths + 2 * padding - kernel_size, stride, rounding_mode="floor") + 1


def _lengths_to_valid_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _build_key_padding_mask(valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(
        (valid_mask.size(0), 1, 1, valid_mask.size(1)),
        device=valid_mask.device,
        dtype=dtype,
    )
    return mask.masked_fill(~valid_mask[:, None, None, :], torch.finfo(dtype).min)


class WhisperIMUMultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.query(x)
        k = self.key(x if xa is None else xa)
        v = self.value(x if xa is None else xa)
        return self.out(self.qkv_attention(q, k, v, mask))

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        q = q.view(n_batch, n_ctx, self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), self.n_head, -1).permute(0, 2, 1, 3)

        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        if mask is not None:
            qk = qk + mask
        weights = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (weights @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class WhisperIMUResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = WhisperIMUMultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = WhisperIMUMultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp),
            nn.GELU(),
            Linear(n_mlp, n_state),
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_ln(x), mask=self_mask)
        if self.cross_attn is not None and xa is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa=xa, mask=cross_mask)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class WhisperIMUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        kernel_size: int = 11,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.n_ctx = n_ctx
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv1 = Conv1d(input_dim, n_state, kernel_size=kernel_size, padding=self.padding)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=kernel_size, stride=2, padding=self.padding)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks: Iterable[WhisperIMUResidualAttentionBlock] = nn.ModuleList(
            [WhisperIMUResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[Tensor, torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)

        output_lengths = _conv_output_length(lengths, kernel_size=self.kernel_size, stride=2, padding=self.padding)
        if x.size(1) > self.n_ctx:
            raise ValueError(
                f"Encoder sequence length {x.size(1)} exceeds n_audio_ctx={self.n_ctx}. "
                "Increase n_audio_ctx or shorten/pad-trim inputs."
            )

        positional = self.positional_embedding[: x.size(1)].to(x.dtype)
        x = x + positional

        valid_mask = _lengths_to_valid_mask(output_lengths, x.size(1))
        attn_mask = _build_key_padding_mask(valid_mask, x.dtype)
        for block in self.blocks:
            x = block(x, self_mask=attn_mask)
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)

        x = self.ln_post(x)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
        return x, output_lengths, valid_mask


class WhisperIMUDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks: Iterable[WhisperIMUResidualAttentionBlock] = nn.ModuleList(
            [WhisperIMUResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-torch.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        audio_features: Tensor,
        audio_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if tokens.size(1) > self.n_ctx:
            raise ValueError(f"Decoder sequence length {tokens.size(1)} exceeds n_text_ctx={self.n_ctx}")

        x = self.token_embedding(tokens) + self.positional_embedding[: tokens.size(1)]
        x = x.to(audio_features.dtype)

        self_mask = self.mask[: tokens.size(1), : tokens.size(1)].to(x.dtype)[None, None]
        cross_mask = None
        if audio_padding_mask is not None:
            cross_mask = _build_key_padding_mask(audio_padding_mask, x.dtype)

        for block in self.blocks:
            x = block(x, xa=audio_features, self_mask=self_mask, cross_mask=cross_mask)

        x = self.ln(x)
        return (x @ self.token_embedding.weight.to(x.dtype).transpose(0, 1)).float()


class WhisperIMUTextRecognizer(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        n_audio_ctx: int = 256,
        n_audio_state: int = 128,
        n_audio_head: int = 4,
        n_audio_layer: int = 4,
        encoder_kernel_size: int = 11,
        n_text_ctx: int = 20,
        n_text_state: int = 128,
        n_text_head: int = 4,
        n_text_layer: int = 4,
        vocab_size: int = 13,
    ):
        super().__init__()
        dims = ModelDimensions(
            n_mels=input_dim,
            n_audio_ctx=n_audio_ctx,
            n_audio_state=n_audio_state,
            n_audio_head=n_audio_head,
            n_audio_layer=n_audio_layer,
            n_vocab=vocab_size,
            n_text_ctx=n_text_ctx,
            n_text_state=n_text_state,
            n_text_head=n_text_head,
            n_text_layer=n_text_layer,
        )
        self.dims = dims
        self.encoder = WhisperIMUEncoder(
            input_dim=input_dim,
            n_ctx=n_audio_ctx,
            n_state=n_audio_state,
            n_head=n_audio_head,
            n_layer=n_audio_layer,
            kernel_size=encoder_kernel_size,
        )
        self.decoder = WhisperIMUDecoder(
            n_vocab=vocab_size,
            n_ctx=n_text_ctx,
            n_state=n_text_state,
            n_head=n_text_head,
            n_layer=n_text_layer,
        )
        if n_audio_state != n_text_state:
            self.encoder_to_decoder = Linear(n_audio_state, n_text_state)
        else:
            self.encoder_to_decoder = nn.Identity()

    def encode(self, inputs: Tensor, lengths: torch.Tensor) -> Dict[str, Tensor]:
        audio_features, output_lengths, audio_padding_mask = self.encoder(inputs, lengths)
        audio_features = self.encoder_to_decoder(audio_features)
        return {
            "audio_features": audio_features,
            "audio_lengths": output_lengths,
            "audio_padding_mask": audio_padding_mask,
        }

    def decode(
        self,
        tokens: Tensor,
        audio_features: Tensor,
        audio_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.decoder(tokens, audio_features, audio_padding_mask=audio_padding_mask)

    def forward(
        self,
        inputs: Tensor,
        lengths: torch.Tensor,
        tokens: Tensor,
    ) -> Dict[str, Tensor]:
        encoded = self.encode(inputs, lengths)
        logits = self.decode(
            tokens,
            encoded["audio_features"],
            audio_padding_mask=encoded["audio_padding_mask"],
        )
        return {
            **encoded,
            "logits": logits,
        }


DIGIT_TOKENS = "0123456789"
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
WHISPER_IMU_VOCAB: List[str] = [PAD_TOKEN, START_TOKEN, END_TOKEN, *list(DIGIT_TOKENS)]
WHISPER_IMU_PAD_IDX = 0
WHISPER_IMU_START_IDX = 1
WHISPER_IMU_END_IDX = 2
WHISPER_IMU_DIGIT_OFFSET = 3
