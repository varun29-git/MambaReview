import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from einops import rearrange, repeat

@dataclass
class InferenceCache:
    """Holds the autoregressive state for efficient token-by-token generation."""
    conv_state: torch.Tensor  # Shape: (batch, d_inner, d_conv)
    ssm_state: torch.Tensor   # Shape: (batch, nheads, headdim, d_state)

class RMSNorm(nn.Module):
    # Standard root mean square normalization. Keeps features from blowing up.
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

def segsum(x: torch.Tensor) -> torch.Tensor:
    """
    Computes a stable segment sum across chunks. 
    Memory-efficient O(N) formulation using broadcasted subtraction instead of repeating.
    x: (..., chunk_len)
    """
    chunk_len = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    
    # Broadcasted subtraction avoids exploding memory constraints
    x_segsum = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
    
    # Mask out future tokens
    mask = torch.tril(torch.ones(chunk_len, chunk_len, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    State Space Duality (SSD) algorithm using pure PyTorch.
    Forces fp32 accumulation for strict numerical stability.
    Uses natively grouped tensor inputs (ngroups) to prevent redundant VRAM usage.
    """
    assert X.shape[1] % block_len == 0
    ngroups = B.shape[2]

    # Force float32 for critical mathematical stability
    X, A, B, C = [tensor.to(torch.float32) for tensor in (X, A, B, C)]

    # Separate heads into (group, heads_per_group) natively for memory efficiency
    X = rearrange(X, "b l (g h) p -> b l g h p", g=ngroups)
    A = rearrange(A, "b l (g h) -> b l g h", g=ngroups)

    # Group everything into chunks of size `block_len`
    X = rearrange(X, "b (c l) g h p -> b c l g h p", l=block_len)
    B = rearrange(B, "b (c l) g n -> b c l g n", l=block_len)
    C = rearrange(C, "b (c l) g n -> b c l g n", l=block_len)
    A = rearrange(A, "b (c l) g h -> b g h c l", l=block_len)

    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Diagonal blocks (Intra-chunk interactions)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclgn,bcsgn,bghcls,bcsghp->bclghp", C, B, L, X)

    # 2. Intra-chunk states
    decay_states = torch.exp((A_cumsum[:, :, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclgn,bghcl,bclghp->bcghpn", B, decay_states, X)

    # 3. Inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    else:
        # Re-group the cached initial states
        initial_states = rearrange(initial_states.to(torch.float32).unsqueeze(1), "b 1 (g h) p n -> b 1 g h p n", g=ngroups)

    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, :, -1], (1, 0))))
    new_states = torch.einsum("bghzc,bcghpn->bzghpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # Flatten the final state back for the cache
    final_state = rearrange(final_state, "b g h p n -> b (g h) p n")

    # 4. Off-diagonal blocks
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclgn,bcghpn,bghcl->bclghp', C, states, state_decay_out)

    # Merge groups, heads, and chunks
    Y = rearrange(Y_diag + Y_off, "b c l g h p -> b (c l) (g h) p")
    return Y, final_state

def ssd_step(X_t, A_t, B_t, C_t, ssm_state):
    """
    Autoregressive step for sequence length = 1.
    Performs standard recurrent SSM update extremely efficiently with ngroups natively.
    """
    X_t, A_t, B_t, C_t = [x.to(torch.float32) for x in (X_t, A_t, B_t, C_t)]
    ssm_state = ssm_state.to(torch.float32)
    
    ngroups = B_t.shape[1]
    
    X_t = rearrange(X_t, "b (g h) p -> b g h p", g=ngroups)
    A_t = rearrange(A_t, "b (g h) -> b g h", g=ngroups)
    ssm_state = rearrange(ssm_state, "b (g h) p n -> b g h p n", g=ngroups)
    
    A_exp = torch.exp(A_t) # (batch, ngroups, heads_per_group)
    
    # Update state: h_t = A * h_{t-1} + B * X
    ssm_state = ssm_state * A_exp.unsqueeze(-1).unsqueeze(-1) + torch.einsum("bgn,bghp->bghpn", B_t, X_t)
    
    # Compute output: Y = C * h_t
    Y_t = torch.einsum("bgn,bghpn->bghp", C_t, ssm_state)
    
    # Re-merge
    Y_t = rearrange(Y_t, "b g h p -> b (g h) p")
    ssm_state = rearrange(ssm_state, "b g h p n -> b (g h) p n")
    
    return Y_t, ssm_state

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
        self.ngroups = 1 # Number of groups (native sharing across heads)
        
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
        
        # Proper Crusty Initializations
        # A_log represents a linear progression across heads (1, 2, ..., nheads)
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads)) # skip connection
        
        # dt_proj bias must be initialized uniformly between log(0.001) and log(0.1)
        dt_init_std = self.nheads ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cache: Optional[InferenceCache] = None) -> Tuple[torch.Tensor, Optional[InferenceCache]]:
        batch, seqlen, _ = x.shape
        dtype = x.dtype
        
        # Split input into our two branches
        x_branch, z_branch = self.in_proj(x).chunk(2, dim=-1)
        
        # Apply the local convolution, tracking the state cache cleanly using .detach() to plug memory leaks
        if cache is not None:
            if seqlen == 1:
                conv_state = cache.conv_state
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = x_branch.squeeze(1).detach()
                cache.conv_state = conv_state
                
                x_branch = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
                if self.conv1d.bias is not None:
                    x_branch = x_branch + self.conv1d.bias
                x_branch = x_branch.unsqueeze(1)
            else:
                x_branch_t = x_branch.transpose(1, 2)
                cache_update = F.pad(x_branch_t, (self.d_conv - x_branch_t.shape[-1], 0)) if x_branch_t.shape[-1] < self.d_conv else x_branch_t[:, :, -self.d_conv:]
                cache.conv_state.copy_(cache_update.detach())
                x_branch = self.conv1d(x_branch_t)[:, :, :seqlen].transpose(1, 2)
        else:
            x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
            
        x_branch = self.act(x_branch)
        
        # Run our big projection and slice it up into the pieces we need
        x_proj_out = self.x_proj(x_branch)
        X, dt, B, C = x_proj_out.split([
            self.d_inner, self.nheads, self.ngroups * self.d_state, self.ngroups * self.d_state
        ], dim=-1)
        
        # We no longer apply `einops.repeat` to B and C to save VRAM. 
        # The einsums natively handle `ngroups` grouping.
        X = rearrange(X, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        dt = F.softplus(self.dt_proj(dt))
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state)
            
        A = -torch.exp(self.A_log)
        
        X_scaled = X * dt.unsqueeze(-1)
        A_scaled = A.unsqueeze(0).unsqueeze(0) * dt
        
        # Run the magic!
        if cache is not None and seqlen == 1:
            Y, new_ssm_state = ssd_step(
                X_scaled.squeeze(1), A_scaled.squeeze(1), 
                B.squeeze(1), C.squeeze(1), cache.ssm_state
            )
            cache.ssm_state.copy_(new_ssm_state.detach())
            Y = Y.unsqueeze(1)
        else:
            pad_len = (self.chunk_size - (seqlen % self.chunk_size)) % self.chunk_size
            if pad_len > 0:
                pad_X = torch.zeros((batch, pad_len, *X_scaled.shape[2:]), device=x.device, dtype=x.dtype)
                pad_A = torch.zeros((batch, pad_len, *A_scaled.shape[2:]), device=x.device, dtype=x.dtype)
                pad_B = torch.zeros((batch, pad_len, *B.shape[2:]), device=x.device, dtype=x.dtype)
                pad_C = torch.zeros((batch, pad_len, *C.shape[2:]), device=x.device, dtype=x.dtype)
                
                X_scaled = torch.cat([X_scaled, pad_X], dim=1)
                A_scaled = torch.cat([A_scaled, pad_A], dim=1)
                B = torch.cat([B, pad_B], dim=1)
                C = torch.cat([C, pad_C], dim=1)

            init_state = cache.ssm_state if cache is not None else None
            Y, final_state = ssd_minimal_discrete(X_scaled, A_scaled, B, C, self.chunk_size, initial_states=init_state)
            
            if cache is not None:
                cache.ssm_state.copy_(final_state.detach())
                
            if pad_len > 0:
                Y = Y[:, :seqlen, :, :]

        Y = Y.to(dtype)
        
        Y = Y + X * self.D.view(1, 1, self.nheads, 1)
        Y = rearrange(Y, "b l h p -> b l (h p)")
        
        Y = Y * self.act(z_branch)
        out = self.out_proj(Y)
            
        return out, cache

class Mamba2Layer(nn.Module):
    # A standard wrapper that handles the residual connection and normalizes the input 
    def __init__(self, d_model: int, expand: int, headdim: int, d_state: int, chunk_size: int, d_conv: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba2Block(d_model, expand, headdim, d_state, chunk_size, d_conv)
    
    def forward(self, x: torch.Tensor, cache: Optional[InferenceCache] = None) -> Tuple[torch.Tensor, Optional[InferenceCache]]:
        mamba_out, cache = self.mamba(self.norm(x), cache)
        return x + mamba_out, cache

class Mamba2Model(nn.Module):
    # The big boss class. Sets up the word embeddings, stacks all the layers together.
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, expand: int, 
                 headdim: int, d_state: int, chunk_size: int, d_conv: int, 
                 pad_vocab_size_multiple: int = 8, tie_embeddings: bool = True):
        super().__init__()
        
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
            
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.n_layer = n_layer
        self.d_conv = d_conv
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.d_state = d_state
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            Mamba2Layer(d_model, expand, headdim, d_state, chunk_size, d_conv) 
            for _ in range(n_layer)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def allocate_inference_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Dict[int, InferenceCache]:
        """Creates empty cache buffers for perfectly fast autoregressive generation."""
        caches = {}
        for i in range(self.n_layer):
            conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
            ssm_state = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=dtype)
            caches[i] = InferenceCache(conv_state=conv_state, ssm_state=ssm_state)
        return caches

    def forward(self, input_ids: torch.Tensor, caches: Optional[Dict[int, InferenceCache]] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x, cache_out = layer(x, cache)
        return self.lm_head(self.norm_f(x))

if __name__ == "__main__":
    device = "cpu"
    # Create the hardened model
    model = Mamba2Model(
        vocab_size=32000, d_model=64, n_layer=2, expand=2, 
        headdim=32, d_state=16, chunk_size=32, d_conv=4
    ).to(device)
    
    # 1. Test standard parallel forward pass (training style)
    print("Testing parallel forward pass (Training Mode)...")
    dummy_input = torch.randint(0, 32000, (2, 40), device=device)
    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (2, 40, model.vocab_size), "Parallel forward failed."

    # 2. Test autoregressive inference (generation style)
    print("\nTesting autoregressive inference pass (Generation Mode)...")
    batch_size = 2
    caches = model.allocate_inference_cache(batch_size, device)
    
    # Prefill phase (e.g. processing a prompt of length 10)
    prompt = torch.randint(0, 32000, (batch_size, 10), device=device)
    prefill_logits = model(prompt, caches)
    print(f"Prefill logits shape: {prefill_logits.shape} | Cache populated!")
    
    # Generation phase (generating 5 tokens step-by-step)
    generated = []
    current_token = prompt[:, -1:] # start with the last token of prompt
    
    for step in range(5):
        step_logits = model(current_token, caches) # shape (batch, 1, vocab_size)
        # Greedily pick the next token
        next_token = step_logits.argmax(dim=-1)
        generated.append(next_token)
        current_token = next_token
        
    print(f"Successfully generated 5 tokens autoregressively in constant time!")