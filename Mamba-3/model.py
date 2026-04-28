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


class Mamba3SISOBlock(nn.Module):
    """
    First-pass SISO block for the Mamba-3 folder.

    The idea is simple: each expanded channel keeps its own small state vector,
    updates that state across the sequence, and then we mix everything back down
    to the model width at the end.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Split the input into a main branch and a gate branch.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # This depthwise conv gives each channel a little local context before the scan.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # For each token we predict:
        # - one raw dt value
        # - one B vector
        # - one C vector
        self.param_proj = nn.Linear(self.d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A stays channel-specific, with one small state vector per inner channel.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D is the usual skip term that lets the current token pass through directly.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape

        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :seqlen]
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.act(x_branch)

        dt_raw, B, C = self.param_proj(x_branch).split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)

        # h is the running state for every expanded channel.
        h = torch.zeros(
            batch,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )
        y = torch.zeros_like(x_branch)

        for t in range(seqlen):
            x_t = x_branch[:, t]
            dt_t = dt[:, t]
            B_t = B[:, t]
            C_t = C[:, t]

            # We discretize the continuous-time parameters one token at a time.
            A_bar = torch.exp(A * dt_t.unsqueeze(-1))
            B_bar = B_t.unsqueeze(1) * dt_t.unsqueeze(-1)

            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = torch.einsum("b i n, b n -> b i", h, C_t)
            y[:, t] = y_t + self.D * x_t

        # The gate branch decides how much of the scanned signal to keep.
        y = y * self.act(z_branch)
        return self.out_proj(y)


class Mamba3SISOLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba3SISOBlock(d_model, d_state, d_conv, expand)

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

        self.layers = nn.ModuleList(
            [
                Mamba3SISOLayer(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
                for _ in range(config.n_layers)
            ]
        )

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

    We keep the class name here already so the file shape stays stable when we
    come back and add the second variant.
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Mamba-3 MIMO is not implemented yet. We are only wiring SISO for now.")


if __name__ == "__main__":
    config = Mamba3Config(vocab_size=4096)
    model = Mamba3SISOModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    logits = model(input_ids)
    print("Logits shape:", logits.shape)
