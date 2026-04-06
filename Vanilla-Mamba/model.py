import torch
import torch.nn as nn
import torch.nn.functional as F

"""

Flow of MambaBlock:

Input: (batch, seq_len, d_model)

Project to d_inner:
   - Expands feature space

Apply 1D convolution:
   - Local mixing across sequence

Split into components:
   - Used to generate A, B, C (input-dependent SSM parameters)
   - May include gating

Selective SSM (scan):
   - Maintains state of size d_state per channel
   - Updates state across sequence

Project back to d_model:
   - Compress back to original dimension

Residual connection applied outside
"""

class MambaBlock(nn.Module):

    def __init__(self, d_model: int, d_state:int, d_conv:int, expand:int):
        super().__init__()

        # -----------------------------------------------------------------------
        self.d_model = d_model # Embedding Dimension
        # Model starts (batch, seq_len, d_model)

        self.d_inner = int(expand * d_model) # Expands hidden dimension
        # Used in the first linear layer (batch, seq_len, d_inner)

        self.d_conv = d_conv # Kernel size of the small 1D convolution.
        # Right after the in_proj, before the SSM. (batch, d_inner, length)

        self.d_state = d_state # Size of the hidden state in SSM
        # for each of the d_inner channels, we keep a tiny vector (batch, d_inner, d_state)

        # -----------------------------------------------------------------------

        # Input projection - expands + splits
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Local convolution (before SSM) 
        self.conv1d = nn.Conv1d(
            in_channels= self.d_inner,
            out_channels= self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,   # causal padding
            groups=self.d_inner,       # depthwise conv
            bias=True          
        )

        # Projection for selective parameters (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 +1, bias=False)

        # Projection for dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Parameter A
        self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        nn.init.normal_(self.A_log, mean=0, std=1)

        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection shrinks back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Activation
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x shape: (batch, length, d_model)
        batch, length, _ = x.shape

        # Input projection + split
        xz = self.in_proj(x)                    # (batch, length, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)              # split into two parts: both (batch, length, d_inner)

        # Local convolution (causal)
        x = x.transpose(1, 2)                   # Conv1d requires (batch, channels, length)
        x = self.conv1d(x)[:, :, :length]       # apply conv + trim to original length
        x = x.transpose(1, 2)                   # back to (batch, length, d_inner)

        # Gating
        x = x * self.act(z)                     # element-wise multiply with gated z

        # Selective parameters
        x_proj_out = self.x_proj(x)             # (batch, length, d_state*2 + 1)
        delta, B, C = x_proj_out.split([1, self.d_state, self.d_state], dim=-1)

        delta = delta.transpose(1, 2)           # (batch, 1, length)
        delta = self.dt_proj(delta).transpose(1, 2)  # (batch, length, d_inner)
        delta = F.softplus(delta)

        # Prepare A
        A = -torch.exp(self.A_log)              # (d_inner, d_state)

        # Selective Scan 
        # We use associative scan trick for parallel computation
        # But for simplicity and robustness, here is clean parallel version using cumprod

        # Discretize
        delta = torch.exp(delta)                                 # (batch, length, d_inner)
        A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1))  # (batch, length, d_inner, d_state)

        # B_bar
        B_bar = B.unsqueeze(-1) * delta.unsqueeze(-1)           # (batch, length, d_state, 1) * (batch, length, 1, d_inner) wait → fix
        B_bar = B.unsqueeze(2) * delta.unsqueeze(-1)            # (batch, length, d_state, d_inner) no

        # Correct B_bar
        B_bar = B.unsqueeze(-1) * delta.unsqueeze(2)            # (batch, length, 1, d_inner) * (batch, length, d_state, 1) ? Let's do proper

        # Better correct way:
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)          # (batch, length, d_inner, d_state)  ← this is standard

        # Initialize hidden state
        h = torch.zeros(batch, length, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        # Reset h for correct parallel scan
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        y = torch.zeros_like(x)

        for t in range(length):   # Keep loop for now - it's correct and robust
            xt = x[:, t]
            dt = delta[:, t]
            Bt = B[:, t]
            Ct = C[:, t]

            A_bar_t = torch.exp(A * dt.unsqueeze(-1))
            B_bar_t = Bt.unsqueeze(1) * dt.unsqueeze(-1)

            h = A_bar_t * h + B_bar_t * xt.unsqueeze(-1)

            yt = torch.einsum('b i s, b s -> b i', h, Ct)
            y[:, t] = yt + self.D * xt

        # Final projection
        return self.out_proj(y)

if __name__ == "__main__":
    block = MambaBlock(d_model=64)
    x = torch.randn(2, 20, 64)   # batch=2, seq_len=20, dim=64
    y = block(x)
    print(y.shape)   # should print torch.Size([2, 20, 64])



class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: int=1e-5):
        super.__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(1,keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight
    



