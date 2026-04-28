import os
import time
import argparse
import random
import math
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
import importlib.util

from eval_utils import compute_perplexity
from model_configs import MODEL_CONFIGS, TRAIN_CONFIG

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "tokenizer_4k.json")

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
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    return Tokenizer.from_file(TOKENIZER_PATH)

def test_tokenizer(tokenizer):
    test_sentence = "Once upon a time, there was a little girl."
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Tokenizer test: '{test_sentence}' -> {len(encoded.ids)} tokens -> '{decoded}'")
    assert decoded.replace(" ", "") == test_sentence.replace(" ", ""), "Tokenizer round-trip failed!"

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


def build_run_name(model_name: str, run_tag: str | None) -> str:
    return model_name if not run_tag else f"{model_name}_{run_tag}"


def build_lr_schedule(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)

    if total_steps <= warmup_steps:
        return 1.0

    decay_progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["mamba1", "mamba2", "mamba3_siso"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--warmup_multiplier", type=float, default=1.0)
    parser.add_argument("--min_lr_ratio", type=float, default=TRAIN_CONFIG.min_lr_ratio)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--untie_embeddings", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    run_name = build_run_name(args.model_name, args.run_tag)
    print(f"Starting run for {run_name} with seed {args.seed} on {DEVICE}")
    model_cfg = MODEL_CONFIGS[args.model_name]
    lr = TRAIN_CONFIG.lr * args.lr_scale
    warmup_steps = max(1, int(TRAIN_CONFIG.warmup_steps * args.warmup_multiplier))

    # 1. Tokenizer Setup & Diagnostic
    tokenizer = get_tokenizer()
    test_tokenizer(tokenizer)

    # 2. Model Initialization
    if args.model_name == "mamba1":
        module = load_model_class(os.path.join(MODELS_DIR, "Vanilla-Mamba", "model.py"), "mamba1_model")
        model = module.MambaModel(
            vocab_size=TRAIN_CONFIG.vocab_size,
            d_model=model_cfg.d_model,
            n_layers=model_cfg.n_layers,
            d_state=model_cfg.d_state,
            d_conv=model_cfg.d_conv,
            expand=model_cfg.expand,
        )
    elif args.model_name == "mamba2":
        mamba2_path = os.path.join(MODELS_DIR, "Mamba-2", "model.py")
        if not os.path.exists(mamba2_path):
            mamba2_path = os.path.join(MODELS_DIR, "mamba-2", "model.py")
            
        module = load_model_class(mamba2_path, "mamba2_model")
        model = module.Mamba2Model(
            vocab_size=TRAIN_CONFIG.vocab_size,
            d_model=model_cfg.d_model,
            n_layer=model_cfg.n_layers,
            expand=model_cfg.expand,
            headdim=model_cfg.headdim,
            d_state=model_cfg.d_state,
            chunk_size=model_cfg.chunk_size,
            d_conv=model_cfg.d_conv,
            ngroups=model_cfg.ngroups,
            tie_embeddings=not args.untie_embeddings,
        )
    else:
        # Mamba-3 lives behind the same CLI switch pattern as the other models.
        module = load_model_class(os.path.join(MODELS_DIR, "Mamba-3", "model.py"), "mamba3_model")
        config = module.Mamba3Config(
            vocab_size=TRAIN_CONFIG.vocab_size,
            d_model=model_cfg.d_model,
            n_layers=model_cfg.n_layers,
            d_state=model_cfg.d_state,
            d_conv=model_cfg.d_conv,
            expand=model_cfg.expand,
            headdim=model_cfg.headdim,
            tie_embeddings=model_cfg.tie_embeddings,
        )
        model = module.Mamba3SISOModel(config)
        
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: build_lr_schedule(
            step=step,
            total_steps=TRAIN_CONFIG.max_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        ),
    )
    
    if USE_AMP:
        scaler = torch.amp.GradScaler('cuda')
        
    # 3. Setup Logging
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f"{run_name}_metrics.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("step,tokens_seen,train_loss,val_ppl,tps,vram_mb,elapsed_seconds,lr\n")

    train_iter = batch_iterator("train", tokenizer, TRAIN_CONFIG.batch_size, TRAIN_CONFIG.seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    tokens_seen = 0
    last_log_tokens = 0
    total_train_loss = 0.0
    steps_since_log = 0
    train_start_time = time.time()
    
    print("\nTraining...")
    model.train()
    
    for step in range(1, TRAIN_CONFIG.max_steps + 1):
        try:
            inputs, labels = next(train_iter)
        except StopIteration:
            train_iter = batch_iterator("train", tokenizer, TRAIN_CONFIG.batch_size, TRAIN_CONFIG.seq_len)
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
                loss = criterion(logits.view(-1, TRAIN_CONFIG.vocab_size), labels.view(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            logits = model(inputs)
            loss = criterion(logits.view(-1, TRAIN_CONFIG.vocab_size), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG.grad_clip)
            optimizer.step()
            scheduler.step()

        if DEVICE == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            step_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            step_time = max(time.time() - t0, 1e-8)

        # Metrics
        batch_tokens = TRAIN_CONFIG.batch_size * TRAIN_CONFIG.seq_len
        tokens_seen += batch_tokens
        total_train_loss += loss.item()
        steps_since_log += 1
        tps = batch_tokens / max(step_time, 1e-8)
        elapsed_seconds = time.time() - train_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # 4. Fixed Checkpoint Evaluation (Every 2M tokens)
        if tokens_seen - last_log_tokens >= 2_000_000 or step == TRAIN_CONFIG.max_steps:
            avg_train_loss = total_train_loss / max(steps_since_log, 1)
            
            # Run Validation
            val_iter = batch_iterator("validation", tokenizer, TRAIN_CONFIG.batch_size, TRAIN_CONFIG.seq_len)
            val_ppl, _ = compute_perplexity(model, val_iter, DEVICE, max_batches=50, vocab_size=TRAIN_CONFIG.vocab_size, pad_id=PAD_ID)
            
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE == "cuda" else 0.0
            
            print(f"Step {step}/{TRAIN_CONFIG.max_steps} | Tokens: {tokens_seen/1e6:.1f}M | "
                  f"Train Loss: {avg_train_loss:.4f} | Val PPL: {val_ppl:.2f} | "
                  f"TPS: {tps:.0f} | VRAM: {vram_mb:.0f}MB")
                  
            with open(log_file, "a") as f:
                f.write(f"{step},{tokens_seen},{avg_train_loss:.4f},{val_ppl:.4f},{tps:.2f},{vram_mb:.2f},{elapsed_seconds:.2f},{current_lr:.8f}\n")
                
            # Save Checkpoint
            torch.save(model.state_dict(), os.path.join(CHECKPOINTS_DIR, f"{run_name}_best.pt"))
            
            # Reset counters
            last_log_tokens = tokens_seen
            total_train_loss = 0.0
            steps_since_log = 0
            model.train()
            
        # 5. Fast Training Logging (Every 50 steps)
        elif step % 50 == 0:
            avg_train_loss = total_train_loss / max(steps_since_log, 1)
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if DEVICE == "cuda" else 0.0
            
            print(f"Step {step}/{TRAIN_CONFIG.max_steps} | Tokens: {tokens_seen/1e6:.1f}M | "
                  f"Train Loss: {avg_train_loss:.4f} | TPS: {tps:.0f} | VRAM: {vram_mb:.0f}MB")
                  
            with open(log_file, "a") as f:
                f.write(f"{step},{tokens_seen},{avg_train_loss:.4f},N/A,{tps:.2f},{vram_mb:.2f},{elapsed_seconds:.2f},{current_lr:.8f}\n")
                
            total_train_loss = 0.0
            steps_since_log = 0

if __name__ == "__main__":
    main()
