# 2. Transformers in Depth

The transformer is the computational core of every model in the rest of this book: the language model, the vision encoder, the vision-language backbone, and most VLA action heads. This chapter builds it completely — from the attention equation to the inference arithmetic that decides whether a VLA can run on a robot.

By the end you should be able to write a transformer from memory, state the shape of every tensor in it, count its parameters and FLOPs, and predict its latency on given hardware.

## Learning objectives

1. Derive scaled dot-product attention, explain the $\sqrt{d_k}$ factor, masks, and the quadratic cost. ([2.1](attention.md))
2. Implement multi-head attention with the reshape trick; explain cross-attention and grouped-query attention. ([2.2](multi-head.md))
3. Prove that attention is permutation-equivariant and derive sinusoidal and rotary position encodings. ([2.3](positions.md))
4. Assemble a pre-norm block (RMSNorm, SwiGLU, residual stream) and distinguish encoder, decoder, encoder–decoder and prefix-LM. ([2.4](block.md))
5. Train a language model correctly: the shift, loss masking, weight tying, AdamW + warm-up + cosine, mixed precision. ([2.5](training.md))
6. Explain prefill vs decode, implement a KV cache, and estimate decode latency from memory bandwidth. ([2.6](inference.md))
7. Count parameters and training FLOPs, and state what scaling laws say — and do not say — for robotics. ([2.7](scaling.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [2.1](attention.md) | Scaled dot-product attention | Lab 03 |
| [2.2](multi-head.md) | Multi-head attention, cross-attention, GQA | Lab 03 |
| [2.3](positions.md) | Positional encodings: learned, sinusoidal, RoPE | Lab 02 |
| [2.4](block.md) | The transformer block and the residual stream | Lab 04 |
| [2.5](training.md) | Training language models | Lab 04 |
| [2.6](inference.md) | Inference: KV cache, bandwidth, latency | Labs 04–05 |
| [2.7](scaling.md) | Parameters, compute, scaling laws | Labs 04–05 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 03 — Attention from scratch** (`labs/03_attention/`)
- **Lab 04 — Transformer block and TinyGPT** (`labs/04_transformer/`): build, train on the depot corpus, KV cache.
- **Lab 05 — Inside a pretrained language model** (`labs/05_pretrained_lm/`): SmolLM2-135M — configuration, parameter count, hidden states, attention maps, logit lens, decoding strategies, prefill/decode timing.
