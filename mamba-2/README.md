# Mamba-2 Architecture

Welcome to the Mamba-2 folder! This isn't just a slight bump to the original architecture—it's a completely new way of looking at state space models. The code here breaks down the paper's *State Space Duality (SSD)* concept into readable, from-scratch PyTorch components. 

Below is a breakdown of how all these classes interact with one another to give us our final working model.

## The Flow of Data
If you drop a batch of tokens into the model, here is the exact chronological order of how they get processed:

### 1. `Mamba2Model`
* **Purpose**: This is the top-level orchestrator. It manages the vocabulary lookup and runs the full stack.
* **What it does**: When token IDs come in, it first maps them into thick feature vectors using the `nn.Embedding`. It then sends these vectors through a chain of `Mamba2Layer`s. At the very end, it normalizes the result and maps it back into vocabulary predictions (logits) using the `lm_head`.

### 2. `Mamba2Layer`
* **Purpose**: A wrapper to make sure training stays stable.
* **What it does**: Neural networks need guardrails so their activations don't explode to infinity. `Mamba2Layer` applies `RMSNorm` to our vectors to tame them, pushes them into the heavy-lifting `Mamba2Block`, and then adds the original vector back in via a "residual connection". 

### 3. `RMSNorm`
* **Purpose**: Fast, simple normalization.
* **What it does**: It enforces a standard variance across the feature dimension. Mamba drops the traditional LayerNorm for RMSNorm because it is cheaper to compute and works just as well.

### 4. `Mamba2Block`
* **Purpose**: The engine of the car. This is where the actual Mamba-2 logic lives.
* **What it does**: 
  - Splits the input into two paths: a main processing branch (`x`) and a gating branch (`z`).
  - Runs a fast 1D causal convolution to let neighboring tokens share information.
  - Generates our four critical State Space Model parameters (`X, dt, B, C`) simultaneously using one large linear projection layer (a huge speedup over Mamba-1).
  - Feeds these parameters into the mathematically complex SSD algorithm.
  - Multiplies the result by the gate branch and spits it out.

### 5. `ssd_minimal_discrete`
* **Purpose**: The core math of State Space Duality.
* **What it does**: The original Mamba had to loop over every single token sequentially, which is hard to parallelize perfectly. Mamba-2 groups the tokens into "chunks". This function takes those chunks and processes them using Matrix Multiplications—the exact same operations that make Transformers so incredibly fast on GPUs. It calculates interactions *inside* the chunk, figures out what memory state to pass onto the *next* chunk, and combines it all back together.

### 6. `segsum`
* **Purpose**: A small helper function for `ssd_minimal_discrete`.
* **What it does**: When doing cumulative sums inside chunks, floating-point math can get weird or unstable. `segsum` calculates these cumulative segment sums cleanly while applying masks to ensure the model can never peek into the future.
