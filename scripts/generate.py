import os
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
SAMPLES_DIR = os.path.join(REPO_ROOT, "samples")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizer_4k.json")

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


def resolve_checkpoint_path(model_name: str) -> str | None:
    primary_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pt")
    fallback_paths = {
        "mamba1": [os.path.join(MODELS_DIR, "Vanilla-Mamba", "mamba1_best.pt")],
        "mamba2": [os.path.join(MODELS_DIR, "Mamba-2", "mamba2_best.pt")],
        "mamba3_siso": [os.path.join(MODELS_DIR, "Mamba-3", "mamba3_siso_best.pt")],
    }

    candidate_paths = [primary_path] + fallback_paths.get(model_name, [])
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

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
    parser.add_argument("--model_name", type=str, required=True, choices=["mamba1", "mamba2", "mamba3_siso"])
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    if args.model_name == "mamba1":
        module = load_model_class(os.path.join(MODELS_DIR, "Vanilla-Mamba", "model.py"), "mamba1_model")
        model = module.MambaModel(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
            d_state=D_STATE, d_conv=D_CONV, expand=EXPAND
        )
    elif args.model_name == "mamba2":
        mamba2_path = os.path.join(MODELS_DIR, "Mamba-2", "model.py")
        if not os.path.exists(mamba2_path):
            mamba2_path = os.path.join(MODELS_DIR, "mamba-2", "model.py")
            
        module = load_model_class(mamba2_path, "mamba2_model")
        model = module.Mamba2Model(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYERS,
            expand=EXPAND, headdim=HEADDIM, d_state=D_STATE,
            chunk_size=CHUNK_SIZE, d_conv=D_CONV, ngroups=NGROUPS
        )
    else:
        module = load_model_class(os.path.join(MODELS_DIR, "Mamba-3", "model.py"), "mamba3_model")
        config = module.Mamba3Config(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            d_state=D_STATE,
            d_conv=D_CONV,
            expand=EXPAND,
            headdim=HEADDIM,
        )
        model = module.Mamba3SISOModel(config)
        
    ckpt_path = resolve_checkpoint_path(args.model_name)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"Loaded {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found for {args.model_name}. Generating with untrained weights.")
        
    model = model.to(DEVICE)
    
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    out_file = os.path.join(SAMPLES_DIR, f"{args.model_name}_step{args.step}.md")
    
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
