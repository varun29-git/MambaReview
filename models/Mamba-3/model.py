import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Mamba3Config:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    is_outproj_norm: bool = False
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class GatedRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * F.silu(gate)


class StateRMSNorm(nn.Module):
    def __init__(self, d_state: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_state))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def apply_rotary_pairs(x: torch.Tensor, phase: torch.Tensor, rotate_dim: int) -> torch.Tensor:
    """
    Apply RoPE-style pairwise rotations to the leading slice of the state dimension.

    x:     (batch, seqlen, nheads, d_state)
    phase: (batch, seqlen, nheads, num_pairs)
    """
    if rotate_dim == 0:
        return x

    x_rot = x[..., :rotate_dim]
    x_pass = x[..., rotate_dim:]

    x_rot = x_rot.reshape(*x_rot.shape[:-1], rotate_dim // 2, 2)
    cos = torch.cos(phase).unsqueeze(-1)
    sin = torch.sin(phase).unsqueeze(-1)

    real = x_rot[..., 0:1]
    imag = x_rot[..., 1:2]

    rotated_real = real * cos - imag * sin
    rotated_imag = real * sin + imag * cos
    rotated = torch.cat([rotated_real, rotated_imag], dim=-1).reshape(*x.shape[:-1], rotate_dim)

    if x_pass.numel() == 0:
        return rotated
    return torch.cat([rotated, x_pass], dim=-1)


class Mamba3SISOBlock(nn.Module):
    """
    Pure PyTorch Mamba-3 SISO block.

    This follows the paper's SISO path instead of the earlier placeholder:
    - no short causal conv
    - token-wise A / dt / trap parameters
    - exponential-trapezoidal recurrence
    - complex state tracking via cumulative RoPE-style rotations on B and C

    The official code uses fused kernels for speed. Here we spell the recurrence
    out directly so it is easy to read and modify.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        rope_fraction: float,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        A_floor: float,
        is_outproj_norm: bool,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.d_inner = d_model * expand
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm

        if self.d_inner % headdim != 0:
            raise ValueError(f"d_inner={self.d_inner} must be divisible by headdim={headdim}")
        self.nheads = self.d_inner // headdim

        if rope_fraction not in (0.5, 1.0):
            raise ValueError("rope_fraction must be 0.5 or 1.0 to match the official Mamba-3 setup.")

        rotate_dim = int(d_state * rope_fraction)
        if rotate_dim % 2 != 0:
            rotate_dim -= 1
        self.rotate_dim = max(rotate_dim, 0)
        self.num_rope_angles = self.rotate_dim // 2

        # The paper version folds every token's recurrence parameters into one projection.
        # We keep the same split so the code maps cleanly to the official implementation.
        in_proj_dim = 2 * self.d_inner + 2 * d_state + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, in_proj_dim, bias=False)

        # Mamba-3 keeps a learned bias for dt and predicts the token-local offset on top.
        dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        dt_bias = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(dt_bias)

        # B and C start from normalized shared content, then get head-specific learned offsets.
        self.B_norm = StateRMSNorm(d_state)
        self.C_norm = StateRMSNorm(d_state)
        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))

        self.D = nn.Parameter(torch.ones(self.nheads))

        if is_outproj_norm:
            self.out_norm = GatedRMSNorm(self.d_inner)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape
        proj = self.in_proj(x)

        split_sizes = [
            self.d_inner,
            self.d_inner,
            self.d_state,
            self.d_state,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]
        z, v, B_raw, C_raw, dd_dt, dd_A, trap_logits, angle_logits = proj.split(split_sizes, dim=-1)

        z = z.view(batch, seqlen, self.nheads, self.headdim)
        v = v.view(batch, seqlen, self.nheads, self.headdim)

        # The shared B/C content is normalized first, then a head-specific bias is added.
        B = self.B_norm(B_raw).unsqueeze(2).expand(-1, -1, self.nheads, -1)
        C = self.C_norm(C_raw).unsqueeze(2).expand(-1, -1, self.nheads, -1)
        B = B + self.B_bias.view(1, 1, self.nheads, self.d_state)
        C = C + self.C_bias.view(1, 1, self.nheads, self.d_state)

        # dt and A are token-dependent in Mamba-3. We do the recurrence in fp32 for stability.
        dt = F.softplus(dd_dt.float() + self.dt_bias.view(1, 1, self.nheads))
        A = -F.softplus(dd_A.float())
        A = torch.clamp(A, max=-self.A_floor)
        trap = torch.sigmoid(trap_logits.float())

        # The complex-valued SSM is realized with cumulative pairwise rotations.
        if self.num_rope_angles > 0:
            phase = angle_logits.float().unsqueeze(2).expand(-1, -1, self.nheads, -1)
            phase = torch.cumsum(phase, dim=1)
            B = apply_rotary_pairs(B.float(), phase, self.rotate_dim)
            C = apply_rotary_pairs(C.float(), phase, self.rotate_dim)
        else:
            B = B.float()
            C = C.float()

        # State layout is (batch, head, head_channel, state_dim).
        state = torch.zeros(
            batch,
            self.nheads,
            self.headdim,
            self.d_state,
            device=x.device,
            dtype=torch.float32,
        )
        prev_k = torch.zeros(batch, self.nheads, self.d_state, device=x.device, dtype=torch.float32)
        prev_v = torch.zeros(batch, self.nheads, self.headdim, device=x.device, dtype=torch.float32)
        outputs = []

        for t in range(seqlen):
            alpha_t = torch.exp(A[:, t] * dt[:, t])
            beta_t = (1.0 - trap[:, t]) * dt[:, t] * alpha_t
            gamma_t = trap[:, t] * dt[:, t]

            k_t = B[:, t]
            q_t = C[:, t]
            v_t = v[:, t].float()

            prev_outer = prev_v.unsqueeze(-1) * prev_k.unsqueeze(-2)
            curr_outer = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)

            state = (
                alpha_t.unsqueeze(-1).unsqueeze(-1) * state
                + beta_t.unsqueeze(-1).unsqueeze(-1) * prev_outer
                + gamma_t.unsqueeze(-1).unsqueeze(-1) * curr_outer
            )

            y_t = torch.einsum("bhpn,bhn->bhp", state, q_t)
            y_t = y_t + self.D.view(1, self.nheads, 1) * v_t
            outputs.append(y_t)

            prev_k = k_t
            prev_v = v_t

        y = torch.stack(outputs, dim=1).to(v.dtype)
        y = y.reshape(batch, seqlen, self.d_inner)
        z = z.reshape(batch, seqlen, self.d_inner)

        if self.is_outproj_norm:
            y = self.out_norm(y, z)
        else:
            y = y * F.silu(z)

        return self.out_proj(y)


class Mamba3SISOLayer(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mamba = Mamba3SISOBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            rope_fraction=config.rope_fraction,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            A_floor=config.A_floor,
            is_outproj_norm=config.is_outproj_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class Mamba3SISOModel(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()

        vocab_size = config.vocab_size
        if config.pad_vocab_size_multiple > 1:
            remainder = vocab_size % config.pad_vocab_size_multiple
            if remainder != 0:
                vocab_size += config.pad_vocab_size_multiple - remainder

        self.config = config
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList([Mamba3SISOLayer(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        return self.lm_head(x)


class Mamba3MIMOModel(nn.Module):
    """
    Placeholder for the later MIMO version.

    The SISO path above is now real Mamba-3 logic. We leave the MIMO class here
    so the folder shape stays stable while we build the second variant later.
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Mamba-3 MIMO is not implemented yet. We are only wiring SISO for now.")


Mamba3Model = Mamba3SISOModel


if __name__ == "__main__":
    config = Mamba3Config(vocab_size=4096)
    model = Mamba3SISOModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(input_ids)
    print("Logits shape:", logits.shape)
