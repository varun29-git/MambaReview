import os
import sys
import torch
import torch.nn as nn
import math
import importlib.util
from datasets import load_dataset
from tokenizers import Tokenizer
import csv

# Config
BATCH_SIZE = 16
SEQ_LEN = 256
VOCAB_SIZE = 4096
EVAL_STEPS = 50

D_MODEL = 256
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2

# Mamba-2 specific configs
HEADDIM = 64
CHUNK_SIZE = 64
NGROUPS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    DEVICE = "mps"

# Special token IDs
BOS_ID = 2
EOS_ID = 3
PAD_ID = 1

def load_model_class(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_tokenizer():
    tokenizer_path = "tokenizer_4k.json"
    return Tokenizer.from_file(tokenizer_path)

def batch_iterator(tokenizer, batch_size, seq_len):
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    buffer = []
    batch_inputs = []
    batch_labels = []

    for example in dataset:
        tokens = [BOS_ID] + tokenizer.encode(example["text"]).ids + [EOS_ID]
        buffer.extend(tokens)

        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len + 1:]

            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            
            batch_inputs.append(input_ids)
            batch_labels.append(labels)

            if len(batch_inputs) == batch_size:
                yield (
                    torch.stack(batch_inputs).to(DEVICE),
                    torch.stack(batch_labels).to(DEVICE),
                )
                batch_inputs = []
                batch_labels = []

@torch.no_grad()
def evaluate_ppl(model, tokenizer):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss = 0.0
    
    iterator = batch_iterator(tokenizer, BATCH_SIZE, SEQ_LEN)
    
    print(f"Evaluating Perplexity for {EVAL_STEPS} steps...")
    for step, (inputs, labels) in enumerate(iterator):
        if step >= EVAL_STEPS:
            break
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
        total_loss += loss.item()
        
    avg_loss = total_loss / EVAL_STEPS
    ppl = math.exp(avg_loss)
    return ppl

@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_tokens=100):
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([BOS_ID] + encoded.ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # If Mamba2, we could use cache, but standard forward works universally
    for _ in range(max_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == EOS_ID:
            break
            
    return tokenizer.decode(input_ids[0].tolist())

def main():
    tokenizer = get_tokenizer()
    
    mamba1_module = load_model_class(os.path.join(os.path.dirname(__file__), "Vanilla-Mamba", "model.py"), "mamba1_model")
    mamba2_module = load_model_class(os.path.join(os.path.dirname(__file__), "Mamba-2", "model.py"), "mamba2_model")

    models = {
        "Mamba-1": mamba1_module.MambaModel(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
            d_state=D_STATE, d_conv=D_CONV, expand=EXPAND
        ),
        "Mamba-2": mamba2_module.Mamba2Model(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYERS,
            expand=EXPAND, headdim=HEADDIM, d_state=D_STATE,
            chunk_size=CHUNK_SIZE, d_conv=D_CONV, ngroups=NGROUPS
        )
    }
    
    # Try to load best checkpoints if they exist
    checkpoints = {
        "Mamba-1": "Vanilla-Mamba/checkpoints/best.pt",
        "Mamba-2": "Mamba-2/checkpoints/best.pt"
    }
    
    results = {}
    
    prompts = [
        "Once upon a time, there was a little",
        "The big red dog decided to",
        "Alice looked at the magical tree and",
        "A small boy named Tim",
        "In the middle of the dark forest,"
    ]
    
    samples_output = "# Zero-Shot Greedy Completions\n\n"
    
    for name, model in models.items():
        print(f"\n--- Processing {name} ---")
        model = model.to(DEVICE)
        ckpt_path = checkpoints[name]
        
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        else:
            print(f"No checkpoint found at {ckpt_path}. Using untrained weights.")
            
        ppl = evaluate_ppl(model, tokenizer)
        print(f"{name} Perplexity: {ppl:.2f}")
        results[name] = ppl
        
        samples_output += f"## {name}\n\n"
        for prompt in prompts:
            comp = greedy_generate(model, tokenizer, prompt)
            samples_output += f"**Prompt**: {prompt}\n\n**Completion**: {comp}\n\n"
            
    with open("samples.md", "w") as f:
        f.write(samples_output)
        
    with open("quality_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Perplexity"])
        for name, ppl in results.items():
            writer.writerow([name, f"{ppl:.2f}"])
            
    print("\nEvaluations complete. Results written to samples.md and quality_metrics.csv")

if __name__ == "__main__":
    main()
