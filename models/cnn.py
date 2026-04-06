"""
Here is a very simple 1D CNN model with residual connections
Most of the code here is dedicated to data augmentation and regularization
Scroll to the bottom to see the model itself
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUAugment(nn.Module):
    """
    CUDA-friendly data augmentation that can be plugged into any PyTorch model
    Does NOT require a lot of extra training time, unlike CPU-based data augmentation :)
    Adds random jitters, scales the signal randomly, rotates around the X axis to simulate rotation of the ring when the user puts it on, and warps the time axis

    Physical Intuition:
    - jitter: sensor noise
    - scaling: different users have different strength
    - rotation: when putting on the ring, the sensor may not always point straight up
    - time warping: different users perform the same gesture at different speed / pace

    Input:  x of shape [B, T, 6] where channels = [ax, ay, az, gx, gy, gz]
    Output: same shape, augmented (only when self.training == True)
    """
    def __init__(
        self,
        p_jitter: float = 0.0,                          # DOES NOT WORK VERY WELL, maybe tune std?
        acc_jitter_std: float = 0.02,                   # noise std in m/s^2
        gyro_jitter_std: float = 0.02,                  # noise std in rad/s
        p_scale: float = 0.8,
        scale_range: tuple[float, float] = (0.9, 1.1),  # multiplicative
        p_rot: float = 0.8,
        max_rot_deg: float = 15.0,                      # rotation around x-axis
        p_timewarp: float = 0.8,
        timewarp_strength: float = 0.3,                 # local speed sampled from N(1, std)
        timewarp_knots: int = 3,                        # number of control points for speed curve
    ):
        super().__init__()
        self.p_jitter = p_jitter
        self.acc_jitter_std = acc_jitter_std
        self.gyro_jitter_std = gyro_jitter_std
        self.p_scale = p_scale
        self.scale_min, self.scale_max = scale_range
        self.p_rot = p_rot
        self.max_rot_rad = max_rot_deg * math.pi / 180.0
        self.p_timewarp = p_timewarp
        self.timewarp_strength = float(timewarp_strength)
        self.timewarp_knots = int(timewarp_knots)


    def _time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time-warp each sample by building a smooth positive speed curve,
        integrating it to get a monotonic source-time mapping, then resampling.

        x: [B, T, C]
        returns: [B, T, C]
        """
        B, T, C = x.shape
        device, dtype = x.device, x.dtype

        # Need at least 2 timesteps to resample meaningfully
        if T < 2:
            return x

        K = max(2, self.timewarp_knots)

        # Sample speed control points around 1.0, then clamp positive.
        # speed in [1-strength, 1+strength] (approximately)
        # lo = max(1e-3, 1.0 - self.timewarp_strength)
        # hi = 1.0 + self.timewarp_strength
        # speed_knots = torch.empty(B, 1, K, device=device, dtype=dtype).uniform_(lo, hi)
        speed_knots = torch.empty(B, 1, K, device=device, dtype=dtype).normal_(1.0, self.timewarp_strength).clamp(min=1e-3)

        # Interpolate speed curve to length T: [B,1,T]
        speed = F.interpolate(speed_knots, size=T, mode="linear", align_corners=True)

        # Integrate speed to get monotonic "source time" curve (unnormalized)
        # src_raw[b, t] increases with t
        src_raw = torch.cumsum(speed.squeeze(1), dim=-1)  # [B,T]

        # Normalize so that src spans [0, T-1]
        src_raw = src_raw - src_raw[:, :1]               # start at 0
        denom = src_raw[:, -1:].clamp_min(1e-6)
        src = src_raw / denom * (T - 1)                  # [B,T] in [0, T-1]

        # Convert to grid_sample coords in [-1, 1] along W dimension
        x_coord = (2.0 * src / (T - 1)) - 1.0            # [B,T]
        y_coord = torch.zeros(B, T, device=device, dtype=dtype)

        # grid: [B, H_out=1, W_out=T, 2] with (x, y)
        grid = torch.stack([x_coord, y_coord], dim=-1).unsqueeze(1)  # [B,1,T,2]

        # Prepare input for 2D grid_sample: [B,C,1,T]
        x_2d = x.permute(0, 2, 1).unsqueeze(2)

        # Sample: output [B,C,1,T]
        # padding_mode doesn't matter much since src is in-range, but "border" is safe.
        warped = F.grid_sample(
            x_2d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Back to [B,T,C]
        return warped.squeeze(2).permute(0, 2, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (x.numel() == 0):
            return x
        assert x.ndim == 3 and x.size(-1) == 6, "Expected [B, T, 6]"

        B, T, C = x.shape
        device, dtype = x.device, x.dtype

        out = x

        # ----- Random jitter (Gaussian noise) -----
        if self.p_jitter > 0:
            do = (torch.rand(B, 1, 1, device=device) < self.p_jitter).to(dtype)
            noise_acc = torch.randn(B, T, 3, device=device, dtype=dtype) * self.acc_jitter_std
            noise_gyr = torch.randn(B, T, 3, device=device, dtype=dtype) * self.gyro_jitter_std
            noise = torch.cat([noise_acc, noise_gyr], dim=-1)  # [B,T,6]
            out = out + do * noise

        # ----- Random scaling (multiplicative) -----
        # Here: scale accel and gyro together per-sample (one factor per sample).
        # If you prefer separate factors for accel/gyro, see comment below.
        if self.p_scale > 0:
            do = (torch.rand(B, 1, 1, device=device) < self.p_scale).to(dtype)
            s = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(self.scale_min, self.scale_max)
            out = out * (1.0 + do * (s - 1.0))

            # Alternative (separate accel & gyro scales):
            # s_ag = torch.empty(B, 1, 2, device=device, dtype=dtype).uniform_(self.scale_min, self.scale_max)
            # s6 = torch.cat([s_ag[..., :1].expand(-1, 1, 3), s_ag[..., 1:].expand(-1, 1, 3)], dim=-1)
            # out = out * (1.0 + do * (s6 - 1.0))

        # ----- Random rotation around x-axis -----
        # Apply same rotation to accel and gyro 3-vectors.
        if self.p_rot > 0:
            do = (torch.rand(B, 1, 1, device=device) < self.p_rot).to(dtype)  # [B,1,1]
            angles = torch.empty(B, 1, 1, device=device, dtype=dtype).uniform_(-self.max_rot_rad, self.max_rot_rad)
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)

            a = out[..., 0:3]
            g = out[..., 3:6]

            # Rotate (y,z) components: y' = c*y - s*z, z' = s*y + c*z ; x unchanged
            ay, az = a[..., 1:2], a[..., 2:3]
            gy, gz = g[..., 1:2], g[..., 2:3]

            ay2 = cos_a * ay - sin_a * az
            az2 = sin_a * ay + cos_a * az
            gy2 = cos_a * gy - sin_a * gz
            gz2 = sin_a * gy + cos_a * gz

            a_rot = torch.cat([a[..., 0:1], ay2, az2], dim=-1)
            g_rot = torch.cat([g[..., 0:1], gy2, gz2], dim=-1)
            rot_out = torch.cat([a_rot, g_rot], dim=-1)

            out = do * rot_out + (1.0 - do) * out

        # ----- Random time warping -----
        if self.p_timewarp > 0:
            do = (torch.rand(B, 1, 1, device=device) < self.p_timewarp).to(dtype)  # [B,1,1]
            warped = self._time_warp(out)
            out = do * warped + (1.0 - do) * out

        return out
    


class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath (per-sample).
    Works for tensors shaped (B, ...). For 1D conv outputs (B, C, L),
    the same mask is broadcast across (C, L).
    """
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Per-sample mask; broadcast across all non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)

        if self.scale_by_keep and keep_prob > 0.0:
            mask = mask / keep_prob

        return x * mask


class ECA1D(nn.Module):
    """
    Efficient Channel Attention (ECA) for 1D CNNs.

    Input:  (B, C, L)
    Output: (B, C, L)

    Paper idea: global average pool -> 1D conv over channels -> sigmoid -> reweight channels.
    """
    def __init__(self, channels: int, k_size: int | None = None, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be > 0")

        # If k_size not provided, use the paper's adaptive kernel size rule
        if k_size is None:
            t = int(abs((math.log2(channels) / gamma) + b))
            k_size = t if t % 2 else t + 1
            k_size = max(k_size, 3)  # keep it odd and >= 3

        if k_size % 2 == 0:
            raise ValueError("k_size must be odd")

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (B, C, L), got {tuple(x.shape)}")

        y = self.avg_pool(x)          # (B, C, 1)
        y = y.transpose(1, 2)         # (B, 1, C)  treat channels as "length" for conv
        y = self.conv(y)              # (B, 1, C)
        y = self.sigmoid(y)
        y = y.transpose(1, 2)         # (B, C, 1)

        return x * y                  # broadcast along L


class ConvBNBlock(nn.Module):
    """
    It does what it says in the name - Conv1d + BN + Activation
    The arguments, except for `norm`, have identical definitions to nn.Conv1d
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int | str = 'same', norm: bool = True):
        super().__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels) if norm else nn.Identity(),
            # nn.ReLU(),
            nn.LeakyReLU(0.1),
            ECA1D(channels=out_channels)
        )
    
    def forward(self, x: torch.Tensor):
        return self.conv_bn(x)
    

class ResidualBlock(nn.Module):
    """
    It does what it says in the name - a residual block with DropPath baked in
    The arguments, except for `norm` and `drop_path`, have identical definitions to nn.Conv1d
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int | str = 'same', norm: bool = True, drop_path: float = 0.0):
        super().__init__()
        self.conv_bn = ConvBNBlock(in_channels, out_channels, kernel_size, stride, padding, norm)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor):
        return x + self.drop_path(self.conv_bn(x))


class SimpleCNN(nn.Module):
    """
    A very simple CNN built with ConvBN blocks. Works incredibly well at around 1.0 MMACs per 200 * 6 input sample
    """
    def __init__(self, input_dim: int, output_dim: int, conv_dim: int, input_dropout1d: float = 0.0, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            # further data augmentation with DRIP
            nn.Dropout1d(input_dropout1d),                                             

            # backbone
            ConvBNBlock(input_dim, conv_dim, 3, 1, 1),                                      
            nn.MaxPool1d(2, 2),                         # length: 200 -> 100
            ConvBNBlock(conv_dim, conv_dim, 3, 1, 1),
            nn.MaxPool1d(2, 2),                         # length: 100 -> 50
            ConvBNBlock(conv_dim, conv_dim, 3, 1, 1),
            nn.MaxPool1d(2, 2),                         # length: 50 -> 25
            ConvBNBlock(conv_dim, conv_dim, 3, 1, 1),
            nn.MaxPool1d(2, 2),                         # length: 25 -> 13
            ConvBNBlock(conv_dim, conv_dim, 3, 1, 1),
            nn.MaxPool1d(2, 2),                         # length: 13 -> 7
            ConvBNBlock(conv_dim, conv_dim, 3, 1, 1),

            # classification head
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(conv_dim, output_dim)
        )
    
    # @jaxtyped(typechecker=beartype)
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.transpose(1, 2)  # B, C, L
        return self.net(x)