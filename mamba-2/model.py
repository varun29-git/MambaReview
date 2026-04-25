import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union

@dataclass
class Mamba2Config:
    """
    Configuration class for Mamba-2 model.
    """
    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: Dict[str, Any] = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    
    # Mamba-2 specific parameters
    headdim: int = 64
    d_state: int = 128
    expand: int = 2
    chunk_size: int = 256
    d_conv: int = 4
    
    def __post_init__(self):
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)


class Mamba2Block(nn.Module):
    """
    The first main structural block for Mamba-2.
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        
        # Placeholder for the actual Mamba-2 layer
        # self.mixer = Mamba2(config)
        # self.norm = nn.RMSNorm(config.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement Mamba-2 SSD logic
        # Returning zeros for now to make the example run without crashing
        return torch.zeros_like(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


class Mamba2Layer(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mamba = Mamba2Block(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        if x is not None:
            x = residual + x
        else:
            x = residual
        return x


class Mamba2Model(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Stack of Mamba-2 layers
        self.layers = nn.ModuleList([
            Mamba2Layer(config) 
            for _ in range(config.n_layer)
        ])
        
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        output: logits of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)          # (batch, seq_len, d_model)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)               # (batch, seq_len, vocab_size)
        
        return logits


if __name__ == "__main__":
    # Example usage
    config = Mamba2Config(
        vocab_size=32000,
        d_model=64,
        n_layer=4,
        d_state=16,
        d_conv=4,
        expand=2
    )
    model = Mamba2Model(config)
    
    input_ids = torch.randint(0, 32000, (2, 20))   # batch=2, seq_len=20
    logits = model(input_ids)
    
    print("Logits shape:", logits.shape)
