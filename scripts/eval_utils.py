import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def compute_perplexity(model, dataloader, device, max_batches=None, vocab_size=4096, pad_id=1):
    """
    Compute perplexity on validation data.
    Returns: (perplexity, avg_loss)
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, (inputs, labels) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
            
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        logits = model(inputs)
        
        # Standard causal language modeling loss computation
        # Shift logits and labels if the dataset iterator didn't already shift them
        # Wait, the dataset iterator in train.py shifts them (input=chunk[:-1], label=chunk[1:])
        # So inputs and labels are already aligned correctly.
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=pad_id
        )
        
        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return float('inf'), float('inf')
        
    avg_loss = total_loss / total_batches
    ppl = math.exp(avg_loss)
    return ppl, avg_loss
