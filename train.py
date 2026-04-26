import os
import time
import argparse
import random
import csv
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
import importlib.util

from eval_utils import compute_perplexity

# Hyperparameters
BATCH_SIZE = 16
SEQ_LEN = 256
VOCAB_SIZE = 4096
MAX_STEPS = 5000
LR = 1e-3
WARMUP_STEPS = 100
GRAD_CLIP = 1.0

# Architecture common
D_MODEL = 256
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2

# Mamba-2 specific
HEADDIM = 64
CHUNK_SIZE = 64
NGROUPS = 1

# System config
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

# Special tokens
BOS_ID = 2
EOS_ID = 3
PAD_ID = 1

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_class(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_tokenizer():
    tokenizer_path = "tokenizer_4k.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    return Tokenizer.from_file(tokenizer_path)

def test_tokenizer(tokenizer):
    test_sentence = "Once upon a time, there was a little girl."
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Tokenizer test: '{test_sentence}' -> {len(encoded.ids)} tokens -> '{decoded}'")
    assert decoded.strip() == test_sentence.strip(), "Tokenizer round-trip failed!"

def batch_iterator(split, tokenizer, batch_size, seq_len):
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
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
                    torch.stack(batch_inputs),
                    torch.stack(batch_labels),
                )
                batch_inputs = []
                batch_labels = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["mamba1", "mamba2"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Starting run for {args.model_name} with seed {args.seed} on {DEVICE}")

    # 1. Tokenizer Setup & Diagnostic
    tokenizer = get_tokenizer()
    test_tokenizer(tokenizer)

    # 2. Model Initialization
    if args.model_name == "mamba1":
        module = load_model_class(os.path.join("Vanilla-Mamba", "model.py"), "mamba1_model")
        model = module.MambaModel(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
            d_state=D_STATE, d_conv=D_CONV, expand=EXPAND
        )
    else:
        module = load_model_class(os.path.join("Mamba-2", "model.py"), "mamba2_model")
        model = module.Mamba2Model(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layer=N_LAYERS,
            expand=EXPAND, headdim=HEADDIM, d_state=D_STATE,
            chunk_size=CHUNK_SIZE, d_conv=D_CONV, ngroups=NGROUPS
        )
        
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    if USE_AMP:
        scaler = torch.amp.GradScaler('cuda')
        
    # 3. Setup Logging
    log_file = os.path.join("logs", f"{args.model_name}_metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("step,tokens_seen,train_loss,val_ppl,tps,vram_mb\n")

    train_iter = batch_iterator("train", tokenizer, BATCH_SIZE, SEQ_LEN)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    tokens_seen = 0
    last_log_tokens = 0
    total_train_loss = 0.0
    steps_since_log = 0
    
    print("\nTraining...")
    model.train()
    
    for step in range(1, MAX_STEPS + 1):
        try:
            inputs, labels = next(train_iter)
        except StopIteration:
            train_iter = batch_iterator("train", tokenizer, BATCH_SIZE, SEQ_LEN)
            inputs, labels = next(train_iter)
            
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        if DEVICE == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.time()

        if USE_AMP:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(inputs)
                loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        if DEVICE == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            step_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            step_time = max(time.time() - t0, 1e-8)

        # Metrics
        batch_tokens = BATCH_SIZE * SEQ_LEN
        tokens_seen += batch_tokens
        total_train_loss += loss.item()
        steps_since_log += 1
        tps = batch_tokens / max(step_time, 1e-8)

        # 4. Fixed Checkpoint Evaluation (Every 2M tokens)
        if tokens_seen - last_log_tokens >= 2_000_000 or step == MAX_STEPS:
            avg_train_loss = total_train_loss / max(steps_since_log, 1)
            
            # Run Validation
            val_iter = batch_iterator("validation", tokenizer, BATCH_SIZE, SEQ_LEN)
            val_ppl, _ = compute_perplexity(model, val_iter, DEVICE, max_batches=50, vocab_size=VOCAB_SIZE, pad_id=PAD_ID)
            
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE == "cuda" else 0.0
            
            print(f"Step {step}/{MAX_STEPS} | Tokens: {tokens_seen/1e6:.1f}M | "
                  f"Train Loss: {avg_train_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                  f"TPS: {tps:.0f} | VRAM: {vram_mb:.0f}MB")
                  
            with open(log_file, "a") as f:
                f.write(f"{step},{tokens_seen},{avg_train_loss:.4f},{val_ppl:.4f},{tps:.2f},{vram_mb:.2f}\n")
                
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join("checkpoints", f"{args.model_name}_best.pt"))
            
            # Reset counters
            last_log_tokens = tokens_seen
            total_train_loss = 0.0
            steps_since_log = 0
            model.train()
            
        # 5. Fast Training Logging (Every 50 steps)
        elif step % 50 == 0:
            avg_train_loss = total_train_loss / max(steps_since_log, 1)
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE == "cuda" else 0.0
            
            print(f"Step {step}/{MAX_STEPS} | Tokens: {tokens_seen/1e6:.1f}M | "
                  f"Train Loss: {avg_train_loss:.4f} | TPS: {tps:.0f} | VRAM: {vram_mb:.0f}MB")
                  
            with open(log_file, "a") as f:
                f.write(f"{step},{tokens_seen},{avg_train_loss:.4f},N/A,{tps:.2f},{vram_mb:.2f}\n")
                
            total_train_loss = 0.0
            steps_since_log = 0

if __name__ == "__main__":
    main()
