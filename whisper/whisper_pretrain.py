from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SPECIAL_PAD_ID = 0
SPECIAL_START_ID = 1
SPECIAL_END_ID = 2
SPECIAL_TOKEN_OFFSET = 3


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int, mode: str = "sincos"):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        if mode not in {"sincos", "zeros"}:
            raise ValueError(f"Unsupported time embedding mode: {mode}")

        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.mode = mode

        if mode == "sincos":
            position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(1e4) / self.d_model)
            )
            encodings = torch.zeros((self.max_len, self.d_model), dtype=torch.float32)
            encodings[:, 0::2] = torch.sin(position * div_term)
            encodings[:, 1::2] = torch.cos(position * div_term)
            self.time_encodings = nn.Parameter(encodings, requires_grad=False)
        else:
            encodings = torch.empty((self.max_len, self.d_model), dtype=torch.float32).uniform_(-0.02, 0.02)
            self.time_encodings = nn.Parameter(encodings, requires_grad=True)

    def forward(self, emb: Tensor) -> Tensor:
        if emb.size(1) > self.max_len:
            raise ValueError(f"Sequence length {emb.size(1)} exceeds max_len={self.max_len}")
        return emb + self.time_encodings[: emb.size(1)].to(dtype=emb.dtype, device=emb.device)


class ConvTokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_filters: Sequence[int] = (128, 128),
        kernel_sizes: Sequence[int] = (7, 7),
        strides: Sequence[int] = (1, 6),
    ):
        super().__init__()
        if not (len(n_filters) == len(kernel_sizes) == len(strides)):
            raise ValueError("n_filters, kernel_sizes, and strides must have the same length.")

        blocks: list[nn.Module] = []
        in_channels = int(input_dim)
        for out_channels, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
            padding = (int(kernel_size) - 1) // 2
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        int(out_channels),
                        kernel_size=int(kernel_size),
                        stride=int(stride),
                        padding=padding,
                    ),
                    nn.GELU(),
                ]
            )
            in_channels = int(out_channels)
        self.out_dim = in_channels
        self.strides = [int(value) for value in strides]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class WhisperPretrainMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.scale = (self.d_model // self.n_heads) ** -0.25

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=True)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=True)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = q.view(*q.shape[:-1], self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:-1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:-1], self.n_heads, -1).permute(0, 2, 1, 3)

        scores = (self.scale * q) @ (self.scale * k).transpose(-1, -2)
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).permute(0, 2, 1, 3).reshape(q.size(0), -1, self.d_model)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_xattn: bool = False):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = WhisperPretrainMultiHeadAttention(d_model, n_heads)
        self.norm_xattn = nn.LayerNorm(d_model) if use_xattn else None
        self.xattn = WhisperPretrainMultiHeadAttention(d_model, n_heads) if use_xattn else None
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        xattn_context: Optional[Tensor] = None,
        xattn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(self.norm_attn(x), self.norm_attn(x), self.norm_attn(x), attn_mask=attn_mask)
        if self.xattn is not None and xattn_context is not None:
            x = x + self.xattn(self.norm_xattn(x), xattn_context, xattn_context, attn_mask=xattn_mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, max_len: int, n_blocks: int, n_heads: int):
        super().__init__()
        self.emb_time = TimeEmbedding(d_model=d_model, max_len=max_len, mode="sincos")
        self.blocks: Iterable[TransformerBlock] = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, use_xattn=False) for _ in range(int(n_blocks))]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x = self.emb_time(x)
        valid_mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_mask = torch.zeros((x.size(0), 1, 1, x.size(1)), device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(~valid_mask[:, None, None, :], torch.finfo(x.dtype).min)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask).masked_fill(~valid_mask.unsqueeze(-1), 0)
        x = self.norm(x).masked_fill(~valid_mask.unsqueeze(-1), 0)
        return x, lengths


class QformerStack(nn.Module):
    def __init__(self, d_model: int, emb_len: int, n_blocks: int, n_heads: int, token_len: int, n_tokens: int):
        super().__init__()
        token_ctx = int(token_len) + 1
        mask = torch.full((token_ctx, token_ctx), float("-inf"), dtype=torch.float32)
        self.register_buffer("attn_mask", torch.triu(mask, diagonal=1), persistent=False)
        self.token_emb = nn.Embedding(int(n_tokens), int(d_model))
        self.emb_time = TimeEmbedding(d_model=int(d_model), max_len=token_ctx, mode="zeros")
        self.blocks: Iterable[TransformerBlock] = nn.ModuleList(
            [TransformerBlock(d_model=int(d_model), n_heads=int(n_heads), use_xattn=True) for _ in range(int(n_blocks))]
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, tokens: Tensor, encoded: Tensor, encoded_lengths: Optional[Tensor]) -> Tensor:
        x = self.emb_time(self.token_emb(tokens))
        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(dtype=x.dtype, device=x.device)[None, None]

        xattn_mask = None
        if encoded_lengths is not None:
            key_padding_mask = torch.arange(encoded.size(1), device=encoded.device).unsqueeze(0) < encoded_lengths.unsqueeze(1)
            xattn_mask = torch.zeros((encoded.size(0), 1, 1, encoded.size(1)), device=encoded.device, dtype=encoded.dtype)
            xattn_mask = xattn_mask.masked_fill(~key_padding_mask[:, None, None, :], torch.finfo(encoded.dtype).min)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, xattn_context=encoded, xattn_mask=xattn_mask)
        x = self.norm(x)
        return (x @ self.token_emb.weight.transpose(0, 1)).float()


class WhisperWithPretrain(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_seq_len: int,
        vocab_size: int,
        token_len: int = 15,
        tokenizer_filters: Sequence[int] = (128, 128),
        tokenizer_kernel_sizes: Sequence[int] = (7, 7),
        tokenizer_strides: Sequence[int] = (1, 6),
        encoder_n_blocks: int = 12,
        encoder_n_heads: int = 8,
        decoder_n_blocks: int = 6,
        decoder_n_heads: int = 4,
        cls_loss_scale: float = 1.0,
    ):
        super().__init__()
        self.tokenizer = ConvTokenizer(
            input_dim=input_dim,
            n_filters=tokenizer_filters,
            kernel_sizes=tokenizer_kernel_sizes,
            strides=tokenizer_strides,
        )
        self.stride_product = math.prod(self.tokenizer.strides)
        emb_len = int(max_seq_len) // self.stride_product
        d_model = int(tokenizer_filters[-1])

        self.encoder = TransformerStack(d_model=d_model, max_len=emb_len, n_blocks=encoder_n_blocks, n_heads=encoder_n_heads)
        self.decoder = QformerStack(
            d_model=d_model,
            emb_len=emb_len,
            n_blocks=decoder_n_blocks,
            n_heads=decoder_n_heads,
            token_len=token_len,
            n_tokens=vocab_size,
        )

        self.pad_id = SPECIAL_PAD_ID
        self.start_id = SPECIAL_START_ID
        self.end_id = SPECIAL_END_ID
        self.cls_loss_scale = float(cls_loss_scale)
        self.max_token_len = int(token_len)

    def encode(self, inputs: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        encoded_init = self.tokenizer(inputs)
        encoded_lengths = lengths // self.stride_product
        return self.encoder(encoded_init, encoded_lengths)

    def forward(
        self,
        inputs: Tensor,
        lengths: Tensor,
        target_tokens: Tensor,
        target_lengths: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        encoded, encoded_lengths = self.encode(inputs, lengths)
        valid_mask = torch.arange(target_tokens.size(1), device=target_tokens.device).unsqueeze(0) < target_lengths.unsqueeze(1)
        y_trg = torch.where(valid_mask, target_tokens, torch.full_like(target_tokens, fill_value=self.pad_id))
        decoder_in = torch.cat(
            [
                torch.full((y_trg.size(0), 1), self.start_id, dtype=y_trg.dtype, device=y_trg.device),
                y_trg,
            ],
            dim=1,
        )
        labels = torch.cat(
            [
                y_trg,
                torch.full((y_trg.size(0), 1), self.pad_id, dtype=y_trg.dtype, device=y_trg.device),
            ],
            dim=1,
        )
        labels = torch.scatter(labels, dim=1, index=target_lengths[:, None], value=self.end_id)
        logits = self.decoder(decoder_in, encoded, encoded_lengths)
        loss_cls = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=self.pad_id)
        return logits[:, :-1], {"total": self.cls_loss_scale * loss_cls, "cls": loss_cls}

    @torch.no_grad()
    def generate(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        encoded, encoded_lengths = self.encode(inputs, lengths)
        tokens = torch.cat(
            [
                torch.full((encoded.size(0), 1), self.start_id, dtype=torch.long, device=encoded.device),
                torch.full((encoded.size(0), self.max_token_len), self.pad_id, dtype=torch.long, device=encoded.device),
            ],
            dim=1,
        )
        predictions = torch.full((encoded.size(0), self.max_token_len), self.end_id, dtype=torch.long, device=encoded.device)
        finished = torch.zeros((encoded.size(0),), dtype=torch.bool, device=encoded.device)

        for step_idx in range(self.max_token_len):
            step_logits = self.decoder(tokens[:, : step_idx + 1], encoded, encoded_lengths)[:, -1]
            step_predictions = step_logits.argmax(dim=-1)
            predictions[:, step_idx] = step_predictions
            tokens[:, step_idx + 1] = step_predictions
            finished |= step_predictions == self.end_id
            if bool(finished.all()):
                break
        return predictions
