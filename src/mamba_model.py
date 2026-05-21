"""
Minimal Mamba (SSM) block — pure PyTorch, CPU-호환.

Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
https://arxiv.org/abs/2312.00752

이 구현은 학습/이해 목적으로 단순화한 버전(mamba-minimal 스타일).
- selective_scan은 직렬 루프로 구현 (CUDA 커널 없음, 작은 시퀀스에서만 실용적)
- 본 캡스톤 데이터(시퀀스 길이 ~30~60)에는 충분히 빠름
- 공식 구현(mamba_ssm 패키지)은 GPU + Triton 필요 — 환경 제약 시 본 구현으로 대체
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int = 64           # 히든 차원
    d_state: int = 16           # SSM 상태 차원 N
    d_conv: int = 4             # 1D conv 커널 크기
    expand: int = 2             # d_inner = expand * d_model
    n_layers: int = 2
    dropout: float = 0.1

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand


class MambaBlock(nn.Module):
    """
    하나의 Mamba 레이어. 입력/출력 모두 (B, L, d_model).

    구조:
        x ──in_proj──→ (x_part, z_part)        # split 2*d_inner → (d_inner, d_inner)
        x_part → conv1d → silu → selective_ssm → y
        y * silu(z_part)  ─ out_proj ─→ output
    """

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.d_inner

        # x, z 동시 생성 (gating)
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_in, bias=False)

        # Depthwise 1D conv over time
        self.conv1d = nn.Conv1d(
            in_channels=d_in,
            out_channels=d_in,
            kernel_size=cfg.d_conv,
            groups=d_in,
            padding=cfg.d_conv - 1,
            bias=True,
        )

        # SSM 파라미터: x → (delta, B, C)
        # delta는 selective time-step, B/C는 선택적 입출력 행렬
        self.x_proj = nn.Linear(d_in, cfg.d_state * 2 + cfg.d_state, bias=False)
        # delta 별도 (low-rank): d_in → 1 → d_in (논문 dt_rank=auto는 d_in/16)
        self.dt_rank = max(1, d_in // 16)
        self.x_proj = nn.Linear(d_in, self.dt_rank + cfg.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_in, bias=True)

        # A: 학습 가능한 (d_inner, d_state). 음수가 되도록 -exp 형태 사용
        A = torch.arange(1, cfg.d_state + 1).float().repeat(d_in, 1)  # (d_in, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_in))  # skip-conn

        # 출력 사영
        self.out_proj = nn.Linear(d_in, cfg.d_model, bias=False)

        # 초기화: dt_proj bias가 softplus(dt_proj.bias) ≈ 0.001~0.1 사이
        dt_init_std = self.dt_rank ** -0.5 * 1.0
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_in) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus inverse
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape
        d_in = self.cfg.d_inner

        xz = self.in_proj(x)                    # (B, L, 2*d_in)
        x_part, z_part = xz.chunk(2, dim=-1)    # 각 (B, L, d_in)

        # Conv over time (depthwise)
        x_conv = x_part.transpose(1, 2)                          # (B, d_in, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]                   # (B, d_in, L)
        x_conv = x_conv.transpose(1, 2)                          # (B, L, d_in)
        x_conv = F.silu(x_conv)

        # SSM core
        y = self._selective_ssm(x_conv)                          # (B, L, d_in)

        # Gating + out
        y = y * F.silu(z_part)
        out = self.out_proj(y)                                   # (B, L, d_model)
        return out

    def _selective_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective state-space scan.
        x: (B, L, d_in)
        return: (B, L, d_in)
        """
        B, L, d_in = x.shape
        N = self.cfg.d_state

        # delta(시간 스텝), B_t, C_t 를 입력 의존적으로 만들기
        proj = self.x_proj(x)                                    # (B, L, dt_rank + 2N)
        dt_low = proj[..., : self.dt_rank]
        Bx = proj[..., self.dt_rank : self.dt_rank + N]          # (B, L, N)
        Cx = proj[..., self.dt_rank + N :]                       # (B, L, N)

        # delta: dt_rank → d_in
        dt = F.softplus(self.dt_proj(dt_low))                    # (B, L, d_in)

        # Discretize A, B
        A = -torch.exp(self.A_log)                               # (d_in, N), negative
        # A_bar[b,l,d,n] = exp(dt[b,l,d] * A[d,n])
        A_bar = torch.exp(dt.unsqueeze(-1) * A)                  # (B, L, d_in, N)
        # B_bar[b,l,d,n] = dt[b,l,d] * Bx[b,l,n]
        B_bar = dt.unsqueeze(-1) * Bx.unsqueeze(2)               # (B, L, d_in, N)

        # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        # 직렬 스캔 (작은 L에서만 실용적)
        h = x.new_zeros(B, d_in, N)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)   # (B, d_in, N)
            # y_t = sum_n C_t[n] * h[..., n]
            y_t = (h * Cx[:, t].unsqueeze(1)).sum(-1)                    # (B, d_in)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                       # (B, L, d_in)

        # Skip connection (D * x)
        y = y + x * self.D
        return y


class MambaForecaster(nn.Module):
    """
    윈도우 기반 시계열 예측기.

    입력: (B, L, F)  — L일치의 F개 feature
    출력: (B, horizon)  — 다음 horizon일의 가격 예측

    구조:
        feature embedding → Linear(F → d_model)
        Stack of MambaBlock × n_layers (residual + LayerNorm)
        last token h_L → Linear(d_model → horizon)
    """

    def __init__(self, n_features: int, horizon: int = 1, cfg: MambaConfig | None = None):
        super().__init__()
        self.cfg = cfg or MambaConfig()
        self.horizon = horizon
        self.embed = nn.Linear(n_features, self.cfg.d_model)
        self.blocks = nn.ModuleList([MambaBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.cfg.d_model) for _ in range(self.cfg.n_layers)])
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.head = nn.Linear(self.cfg.d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, n_features) -> (B, horizon)"""
        h = self.embed(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + self.dropout(block(norm(h)))   # pre-norm + residual
        last = h[:, -1, :]                         # (B, d_model)
        return self.head(last)                     # (B, horizon)


if __name__ == "__main__":
    cfg = MambaConfig(d_model=32, d_state=8, n_layers=2)
    model = MambaForecaster(n_features=10, horizon=1, cfg=cfg)
    x = torch.randn(4, 30, 10)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"입력 {tuple(x.shape)} → 출력 {tuple(y.shape)},  파라미터 {n_params:,}개")
