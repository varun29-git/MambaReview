import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class RMSNorm(nn.Module):
    # Standard root mean square normalization. Keeps features from blowing up.
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

def segsum(x):
    # This computes a stable segment sum across chunks. 
    # Mamba-2 relies heavily on cumulative sums for its internal recurrence, 
    # but we have to mask out future tokens so the model doesn't cheat.
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    # State Space Duality (SSD) algorithm. 
    # Instead of looping token-by-token like a recurrent network, we break the sequence 
    # into blocks and use matrix multiplication (like attention) to process them fast.
    
    # Check if things divide nicely into chunks
    assert X.shape[1] % block_len == 0

    # Group everything into chunks of size `block_len`
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # Step 1: Calculate how tokens inside the same chunk interact with each other (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # Step 2: Figure out the hidden state at the end of each chunk
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # Step 3: Pass that hidden state over to the next chunks so they have memory of the past
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Step 4: Use the past hidden states to influence the current chunk's output (off-diagonal blocks)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Finally, glue it all back together
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state

class Mamba2Block(nn.Module):
    # This is the actual engine of Mamba-2. It takes input, mixes it up locally with a 1D conv,
    # projects it to get our SSM parameters (A, B, C, X), and feeds them into the SSD math.
    def __init__(self, d_model: int, expand: int, headdim: int, d_state: int, chunk_size: int, d_conv: int):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.d_state = d_state
        self.chunk_size = chunk_size
        self.d_conv = d_conv
        self.ngroups = 1 # Number of heads that share the same B/C params
        
        # We project the input into two paths: 
        # the main SSM path (x) and a multiplicative gate path (z)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        
        # A tiny causal convolution just to give tokens a tiny bit of local context before the big SSM step
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=self.d_conv,
            padding=self.d_conv - 1, groups=self.d_inner, bias=True
        )
        
        # Mamba-2 trick: instead of projecting B, C, dt, and X separately, we do it in one big fat projection
        # This makes it way faster on the GPU
        self.x_proj_dim = self.d_inner + self.nheads + 2 * self.ngroups * self.d_state
        self.x_proj = nn.Linear(self.d_inner, self.x_proj_dim, bias=False)
        
        # A small projection for dt (the step size). Needs bias.
        self.dt_proj = nn.Linear(self.nheads, self.nheads, bias=True)
        
        # 'A' is just a learned parameter array now, not input-dependent!
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads)) # skip connection
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = x.shape
        
        # If the sequence isn't a clean multiple of chunk_size, we just pad it with zeros at the end
        pad_len = (self.chunk_size - (seqlen % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            
        padded_seqlen = seqlen + pad_len
        
        # Split input into our two branches
        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)
        
        # Apply the local convolution
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :padded_seqlen]
        x_branch = self.act(x_branch.transpose(1, 2))
        
        # Run our big projection and slice it up into the pieces we need
        x_proj_out = self.x_proj(x_branch)
        X, dt, B, C = x_proj_out.split([
            self.d_inner, 
            self.nheads, 
            self.ngroups * self.d_state, 
            self.ngroups * self.d_state
        ], dim=-1)
        
        # Reshape everything so the dimensions line up for the matrix math
        X = rearrange(X, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        dt = F.softplus(self.dt_proj(dt))
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state)
        
        # Just broadcast B and C to all heads 
        B = repeat(B, "b l 1 n -> b l h n", h=self.nheads)
        C = repeat(C, "b l 1 n -> b l h n", h=self.nheads)
            
        A = -torch.exp(self.A_log)
        
        # Prep the variables for the SSD algorithm. We multiply X and A by our step size dt
        X_scaled = X * dt.unsqueeze(-1)
        A_scaled = A.unsqueeze(0).unsqueeze(0) * dt
        
        # Run the magic!
        Y, _ = ssd_minimal_discrete(X_scaled, A_scaled, B, C, self.chunk_size)
        
        # Add the D skip connection, then squish heads back down into the inner dimension
        Y = Y + X * self.D.view(1, 1, self.nheads, 1)
        Y = rearrange(Y, "b l h p -> b l (h p)")
        
        # Multiply by our gate branch and map back to the original model dimension
        Y = Y * self.act(z_branch)
        out = self.out_proj(Y)
        
        # Chop off any padding we added earlier
        if pad_len > 0:
            out = out[:, :seqlen, :]
            
        return out

class Mamba2Layer(nn.Module):
    # A standard wrapper that handles the residual connection and normalizes the input 
    # before tossing it into the Mamba block
    def __init__(self, d_model: int, expand: int, headdim: int, d_state: int, chunk_size: int, d_conv: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba2Block(d_model, expand, headdim, d_state, chunk_size, d_conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))

class Mamba2Model(nn.Module):
    # The big boss class. Sets up the word embeddings, stacks all the layers together, 
    # and spits out the final token probabilities.
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, expand: int, 
                 headdim: int, d_state: int, chunk_size: int, d_conv: int, 
                 pad_vocab_size_multiple: int = 8, tie_embeddings: bool = True):
        super().__init__()
        
        # Pad the vocab to a nice multiple so hardware feels happy
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
            
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Stack 'em up
        self.layers = nn.ModuleList([
            Mamba2Layer(d_model, expand, headdim, d_state, chunk_size, d_conv) 
            for _ in range(n_layer)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tying weights helps save parameters since the input and output spaces are the same
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm_f(x))


if __name__ == "__main__":
    # A quick dry run to make sure everything links up without exploding
    model = Mamba2Model(
        vocab_size=32000, d_model=64, n_layer=4, expand=2, 
        headdim=32, d_state=16, chunk_size=32, d_conv=4
    )
    
    dummy_input = torch.randint(0, 32000, (2, 20))
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)