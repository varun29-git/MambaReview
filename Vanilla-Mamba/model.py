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

        # Selective parameters (Δ, B, C)
        x_proj_out = self.x_proj(x)                          # (batch, length, d_state*2 + 1)
        delta, B, C = x_proj_out.split([1, self.d_state, self.d_state], dim=-1)
        
        # Reshape and project delta properly
        delta = delta.transpose(1, 2)                        # (batch, 1, length) → for dt_proj
        delta = self.dt_proj(delta).transpose(1, 2)          # (batch, length, d_inner)
        delta = F.softplus(delta)                            # ensure delta > 0

        # Prepare A (decay) - shape (d_inner, d_state)
        A = -torch.exp(self.A_log)                           # negative for stability

        # Selective Scan - Correct & Robust sequential version (easy to debug)
        batch_size, seq_len, _ = x.shape
        y = torch.zeros_like(x)
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for t in range(seq_len):
            xt = x[:, t]                                     # (batch, d_inner)
            dt = delta[:, t]                                 # (batch, d_inner)
            Bt = B[:, t]                                     # (batch, d_state)
            Ct = C[:, t]                                     # (batch, d_state)
            
            # Discretization (zero-order hold, standard in Mamba)
            dt = torch.exp(dt)                               # make dt positive and scaled
            A_bar = torch.exp(A * dt.unsqueeze(-1))         # (batch, d_inner, d_state) via broadcasting
            B_bar = Bt.unsqueeze(1) * dt.unsqueeze(-1)      # (batch, 1, d_state) * (batch, d_inner, 1) wait - fix below
            
            # Correct B_bar broadcasting
            B_bar = (Bt.unsqueeze(1) * dt.unsqueeze(-1))    # (batch, d_inner, d_state) - this is the standard way
            
            # Update hidden state
            h = A_bar * h + B_bar * xt.unsqueeze(-1)
            
            # Output for this timestep
            yt = torch.einsum("b i s, b s -> b i", h, Ct)    # or (h * Ct.unsqueeze(1)).sum(-1)
            y[:, t] = yt + self.D * xt                       # D is the skip connection
            
        # Final output projection
        return self.out_proj(y)


if __name__ == "__main__":
    block = MambaBlock(d_model=64)
    x = torch.randn(2, 20, 64)   # batch=2, seq_len=20, dim=64
    y = block(x)
    print(y.shape)   # should print torch.Size([2, 20, 64])