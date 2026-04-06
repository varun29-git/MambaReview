import torch
import torch.nn as nn
import time
import math
import os

from model import MambaModel

from datasets import load_dataset
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 32
SEQ_LEN = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
MAX_STEPS = 1000         
EVAL_INTERVAL = 200
EVAL_STEPS = 20
LOG_INTERVAL = 50
WARMUP_STEPS = 200
GRAD_CLIP = 1.0
CHECKPOINT_DIR = "checkpoints"

VOCAB_SIZE = 8192         # VectorCLM tokenizer vocab size
D_MODEL = 256
N_LAYERS = 6
D_STATE = 16
D_CONV = 4
EXPAND = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"  # Mixed precision only on GPU

# Special token IDs (from VectorCLM tokenizer)
UNK_ID = 0   # <UNK>
PAD_ID = 1   # <PAD>
BOS_ID = 2   # <BOS>
EOS_ID = 3   # <EOS>


# ---------------------------------------------------------------------------
# Tokenizer 
# ---------------------------------------------------------------------------

def get_tokenizer():
    """Load the custom BPE tokenizer from VectorCLM."""
    tokenizer = Tokenizer.from_file("tokenizer.json")
    return tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_dataloader(split: str, tokenizer, batch_size: int, seq_len: int):
    """Stream TinyStories and yield (input_ids, labels) batches."""

    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    buffer = []

    for example in dataset:
        tokens = [BOS_ID] + tokenizer.encode(example["text"]).ids + [EOS_ID]
        buffer.extend(tokens)

        while len(buffer) >= seq_len + 1:
            chunk = buffer[:seq_len + 1]
            buffer = buffer[seq_len + 1:]

            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            yield input_ids, labels


def batch_iterator(split, tokenizer, batch_size, seq_len):
    """Collate individual samples into batches."""

    batch_inputs = []
    batch_labels = []

    for input_ids, labels in get_dataloader(split, tokenizer, batch_size, seq_len):
        batch_inputs.append(input_ids)
        batch_labels.append(labels)

        if len(batch_inputs) == batch_size:
            yield (
                torch.stack(batch_inputs).to(DEVICE),
                torch.stack(batch_labels).to(DEVICE),
            )
            batch_inputs = []
            batch_labels = []


# ---------------------------------------------------------------------------
# Learning Rate Schedule (cosine with warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    
    decay_ratio = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE * max(coeff, 0.1)


# ---------------------------------------------------------------------------
# Generation (for qualitative checks)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8):
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([BOS_ID] + encoded.ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == EOS_ID:
            break

    model.train()
    return tokenizer.decode(input_ids[0].tolist())


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train():
    print(f"Device: {DEVICE}")

    # Tokenizer
    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # Model
    model = MambaModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Loss (ignore PAD tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # Mixed Precision
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    # Training
    model.train()
    step = 0
    total_loss = 0.0
    best_eval_loss = float("inf")
    start_time = time.time()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    data_iter = batch_iterator("train", tokenizer, BATCH_SIZE, SEQ_LEN)

    print(f"\nStarting training for {MAX_STEPS} steps...")
    print(f"Batch size: {BATCH_SIZE} | Seq length: {SEQ_LEN} | AMP: {USE_AMP}")
    print("-" * 60)

    for input_ids, labels in data_iter:
        if step >= MAX_STEPS:
            break

        # LR schedule
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward (mixed precision)
        with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
            logits = model(input_ids)
            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))

        # Backward (scaled gradients)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        step += 1

        # Log
        if step % LOG_INTERVAL == 0:
            avg_loss = total_loss / LOG_INTERVAL
            elapsed = time.time() - start_time
            tokens_per_sec = (step * BATCH_SIZE * SEQ_LEN) / elapsed
            print(
                f"Step {step:>5d}/{MAX_STEPS} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:.0f} | "
                f"Elapsed: {elapsed / 60:.1f}min"
            )
            total_loss = 0.0

        # Eval
        if step % EVAL_INTERVAL == 0:
            model.eval()
            eval_loss = 0.0
            eval_iter = batch_iterator("validation", tokenizer, BATCH_SIZE, SEQ_LEN)

            with torch.no_grad():
                for eval_step, (eval_input, eval_label) in enumerate(eval_iter):
                    if eval_step >= EVAL_STEPS:
                        break
                    with torch.amp.autocast(device_type=DEVICE, enabled=USE_AMP):
                        eval_logits = model(eval_input)
                        eval_loss += criterion(
                            eval_logits.view(-1, VOCAB_SIZE), eval_label.view(-1)
                        ).item()

            eval_loss /= EVAL_STEPS
            print(f"  >> Eval loss: {eval_loss:.4f}")

            # Save best
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
                print(f"  >> Saved best checkpoint (eval_loss={eval_loss:.4f})")

            # Generate a sample
            sample = generate(model, tokenizer, "Once upon a time", max_new_tokens=80)
            print(f"  >> Sample: {sample[:200]}")
            print("-" * 60)

            model.train()

    # Final save
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.pt"))

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes.")
    print(f"Best eval loss: {best_eval_loss:.4f}")

    # Final generation
    print("\n--- Final Generation Samples ---")
    for prompt in ["Once upon a time", "The little cat", "There was a boy named"]:
        text = generate(model, tokenizer, prompt, max_new_tokens=100)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: {text[:300]}")


if __name__ == "__main__":
    train()
