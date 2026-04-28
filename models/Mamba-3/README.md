# Mamba-3

This directory holds the **Mamba-3** implementation for this repo.

For now we are wiring the **SISO** version into training first. The file layout matches the other model folders in the repository: the main implementation lives in `model.py`, and the folder is ready to grow into the later **MIMO** version without changing that top-level shape.

## What is in `model.py`

- `Mamba3Config`: small config object for model construction.
- `Mamba3SISOBlock`: a pure PyTorch Mamba-3 SISO block with token-wise `A/dt`, trapezoidal recurrence, and RoPE-style complex state tracking.
- `Mamba3SISOLayer`: RMSNorm + residual wrapper around the block.
- `Mamba3SISOModel`: embedding stack, repeated layers, final norm, and LM head.
- `Mamba3MIMOModel`: placeholder for the later MIMO pass.

## Current status

This is now a readable, trainable SISO implementation of the main Mamba-3 ideas in pure PyTorch. It is not the same as the official fused-kernel implementation, so it will be slower, but the underlying recurrence is no longer a placeholder.

The remaining missing piece is the MIMO path.
