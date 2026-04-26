import os
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import importlib.util

# Config
VOCAB_SIZE = 4096
D_MODEL = 256
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2
HEADDIM = 64
CHUNK_SIZE = 64
NGROUPS = 1

BOS_ID = 2
EOS_ID = 3

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

PROMPTS = [
    "Once upon a time",
    "The little cat",
    "There was a boy named",
    "One day, a girl found",
    "In a magical forest"
]

def load_model_class(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def top_p_sampling(logits, top_p=0.9, temperature=0.8):
    if temperature != 1.0:
        logits = logits / temperature
        
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_p=0.9):
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([BOS_ID] + encoded.ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    generated_tokens = []
    
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        generated_tokens.append(next_token.item())
        
        # In this strict task, we want exactly 100 tokens, but we break if EOS
        # The prompt said "Generates exactly 100 tokens per prompt". We will force 100 tokens.
        # But if we hit EOS, maybe we just stop? The user said exactly 100 tokens.
        # We won't break on EOS just to strictly fulfill "exactly 100 tokens".
        
    return tokenizer.decode(input_ids[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["mamba1", "mamba2"])
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer_path = "tokenizer_4k.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    if args.model_name == "mamba1":
        module = load_model_class(os.path.join("Vanilla-Mamba", "model.py"), "mamba1_model")
        model = module.MambaModel(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
            d_state=D_STATE, d_conv=D_CONV, expand=EXPAND
        )
    else:
        mamba2_path = os.path.join("Mamba-2", "model.py")
        if not os.path.exists(mamba2_path):
            mamba2_path = os.path.join("mamba-2", "model.py")
            
        module = load_model_class(mamba2_path, "mamba2_model")
        model = module.Mamba2Model(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYERS,
            expand=EXPAND, headdim=HEADDIM, d_state=D_STATE,
            chunk_size=CHUNK_SIZE, d_conv=D_CONV, ngroups=NGROUPS
        )
        
    ckpt_path = os.path.join("checkpoints", f"{args.model_name}_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"Loaded {ckpt_path}")
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found. Generating with untrained weights.")
        
    model = model.to(DEVICE)
    
    out_file = os.path.join("samples", f"{args.model_name}_step{args.step}.md")
    
    md_content = f"# Samples: {args.model_name.upper()} (Step {args.step})\n\n"
    md_content += f"**Settings**: Temperature = {args.temperature}, Top-P = {args.top_p}\n\n"
    
    for prompt in PROMPTS:
        print(f"Generating for: '{prompt}'...")
        completion = generate_text(model, tokenizer, prompt, max_tokens=100, temperature=args.temperature, top_p=args.top_p)
        md_content += f"## Prompt: {prompt}\n\n"
        md_content += f"```text\n{completion}\n```\n\n"
        
    with open(out_file, "w") as f:
        f.write(md_content)
        
    print(f"Samples saved to {out_file}")

if __name__ == "__main__":
    main()
