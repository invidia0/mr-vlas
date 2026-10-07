# Chapter 2 Review

## Key ideas

1. **Attention** is a differentiable dictionary lookup / kernel regression / message passing on a complete graph: $\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V$, rows normalised over keys. The $\sqrt{d_k}$ keeps the softmax unsaturated; the mask defines who may read whom; cost is quadratic in length.
2. **Multi-head** attention runs $h$ narrow heads in parallel via a reshape; heads read independently and write additively. **Cross-attention** lets one stream read another; **GQA** shrinks the KV cache.
3. Attention is **permutation-equivariant**; order comes from **positional encodings** — learned, sinusoidal, or **RoPE**, which makes scores depend on relative offsets.
4. The **pre-norm block** — RMSNorm → attention → add; RMSNorm → SwiGLU → add — is ~$12d^2$ parameters. The **residual stream** is the model's shared working memory.
5. Training = shifted next-token cross-entropy with a **loss mask**, AdamW, warm-up + cosine, bf16, clipping.
6. Inference = **prefill** (compute-bound) + **decode** (bandwidth-bound) with a **KV cache**; latency ≈ bytes / bandwidth per sequential step.
7. $N\approx12Ld^2$, $C\approx6ND$; scaling laws for language are well established, for robots they are not.

## Exercises

1. **Gradient of attention.** For a single query, derive $\partial\mathbf{y}/\partial\mathbf{q}$ where $\mathbf{y}=\sum_j a_j\mathbf{v}_j$, $a=\operatorname{softmax}(\mathbf{q}K^\top/\sqrt{d_k})$. Show it vanishes when $a$ is one-hot.
2. **Masks.** Write the boolean mask for a VLA sequence `[img (4)] [text (3)] [state (1)] [actions (3)]` where image and text attend bidirectionally among themselves, the state token attends to everything before it and itself, and action tokens attend to everything except later action tokens. Then modify it so that actions attend to each other bidirectionally (as in π0's action block).
3. **Equivariance.** Prove that a stack of pre-norm blocks without positional encodings and without a mask is permutation-equivariant. Which single change breaks it most cheaply?
4. **RoPE.** Show that with RoPE, $\tilde{\mathbf{q}}_m\tilde{\mathbf{k}}_m^\top=\mathbf{q}\mathbf{k}^\top$ (same position). What does this imply about how RoPE treats a token attending to itself?
5. **Parameters.** SmolLM2-135M has $d=576$, $L=30$, $h=9$ query heads, $h_\text{kv}=3$, $d_\text{ff}=1536$ (SwiGLU), $V=49{,}152$, tied embeddings. Compute its parameter count by hand and compare with Lab 05.
6. **Latency.** A flow-matching VLA has a 2B backbone (bf16) run once per step and a 300M action expert run 10 times per step, on a 200 GB/s device. Estimate the lower bound on step latency. Compare with a 7B token-based VLA emitting 7 tokens. Which is faster, and by how much?
7. **Research question.** In a team of 5 drones, each drone's VLA reads its own camera (256 tokens) and the latest message from every other drone. Compare (a) appending the other drones' raw image tokens to each drone's context, and (b) having each drone broadcast 8 cross-attention-compressed summary tokens. Give attention cost per layer and bandwidth per broadcast for both, as functions of team size $N$.

<!-- quiz: ch02 -->

## Reading assignments

1. [Vaswani et al. (2017)](../../papers.md#vaswani2017) — Section 3 in full, with this chapter open. Reproduce Table 1's complexity column.
2. [Elhage et al. (2021)](../../papers.md#elhage2021) — "The residual stream" and "Attention heads are independent and additive".
3. [Su et al. (2021)](../../papers.md#su2021) — Section 3.2.
4. [Dao et al. (2022)](../../papers.md#dao2022) — Section 3.1 (the online softmax).
5. [Kaplan et al. (2020) / Hoffmann et al. (2022)](../../papers.md#scaling-laws) — figures only, then Chinchilla's Section 3.
