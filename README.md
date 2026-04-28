# Mamba Review

**Status: Under Development** 

My aim with this repository is to systematically learn, explore, and implement all the Mamba architectures, starting from the ground up with **Vanilla Mamba**. 

## Foundational Research Papers

This learning journey is guided by the following foundational papers on Mamba and its variants:

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
  *Albert Gu, Tri Dao*  
  [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

- **Transformers are SSMs: Generalized Models and Tensor Fusions (Mamba-2)**  
  *Tri Dao, Albert Gu*  
  [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

- **Mamba-3: Improved Sequence Modeling using State Space Principles**  
  *Albert Gu, Tri Dao*  
  [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)

- **Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model (Vim)**  
  *Lianghui Zhu et al.*  
  [arXiv:2401.09417](https://arxiv.org/abs/2401.09417)

- **VMamba: Visual State Space Model**  
  *Yue Liu et al.*  
  [arXiv:2401.10166](https://arxiv.org/abs/2401.10166)

- **Jamba: A Hybrid Transformer-Mamba Language Model**  
  *AI21 Labs*  
  [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)

## Tokenizer

This repository uses a custom Byte-Pair Encoding (BPE) tokenizer trained specifically on the `roneneldan/TinyStories` dataset. 

In order to maximize parameter efficiency for these small-scale research models, we deliberately restrict the vocabulary size to **4,096 tokens**. By shrinking the embedding matrix, the parameter budget is freed up to focus on the core Mamba architecture blocks. You can generate the tokenizer by running `python3 scripts/train_tokenizer.py`.

## Model Leaderboard

| Model | Throughput (TPS) | Validation PPL |
| :--- | :--- | :--- |
| Vanilla Mamba | N/A | N/A |
| Mamba-2 | 66184 | 16.01 |
| Mamba-3 SISO | N/A | N/A |

