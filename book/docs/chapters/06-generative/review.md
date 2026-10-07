# Chapter 6 Review

## Key ideas

1. **MSE predicts the conditional mean**; for multimodal demonstrations the mean is invalid. Any unimodal likelihood has the same problem. Modes come from symmetric choices, ambiguous instructions, different demonstrators, hidden variables and timing.
2. **MDNs** and **discretised autoregressive** heads write the distribution explicitly and choose the mode first — a mixture component, a first token — which keeps samples coherent. Independent per-dimension bins are *not* coherent (51% invalid in Lab 09).
3. The **ELBO** $=\mathbb{E}_q[\log p(\mathbf{a}\mid\mathbf{z})]-D_\text{KL}(q\Vert p)$ trains latent-variable models; the **reparameterisation trick** makes it differentiable. **Posterior collapse** silently returns a CVAE to regression (Lab 09: already at β = 1). ACT uses a CVAE over 100-step chunks and sets $\mathbf{z}=\mathbf{0}$ at test time.
4. **Diffusion** learns to predict the noise added to an action; its loss comes from the ELBO, its optimum is the scaled score, and sampling removes noise in $K$ steps (DDIM: fewer, deterministic).
5. **Flow matching** regresses the velocity $\mathbf{x}_1-\mathbf{x}_0$ of straight noise-to-data lines; the learned average velocity field transports noise to *each* mode. It is the action model of π0, GR00T N1 and SmolVLA.
6. **Sampling cost** is network evaluations × network size, on top of one backbone pass. Token-based policies pay backbone *decode passes*; diffusion and flow pay passes of a small expert. Chunking trades reactivity for throughput; asynchronous execution removes pauses but adds staleness.

## Exercises

1. **Median.** Show that the L1 loss is minimised by the conditional median. For a symmetric bimodal distribution, where is it?
2. **MDN gradient.** Derive $\partial(-\log p)/\partial\boldsymbol\mu_k$ for an MDN and show it is the Gaussian gradient weighted by the responsibility $r_k$.
3. **Binning error and tokens.** An arm's 7-D action has ranges ±3 cm (position), ±0.2 rad (rotation) and a binary gripper. With 256 bins, what is the worst-case quantisation error in each? How many tokens does a 50-step chunk need without compression?
4. **ELBO gap.** For a model with a true posterior $p(\mathbf{z}\mid\mathbf{a})=\mathcal{N}(1,0.1^2)$ and $q=\mathcal{N}(0,1)$, compute the gap $\log p-\operatorname{ELBO}$.
5. **Diffusion.** Using $\bar\alpha_k$, show that $\mathbf{x}_k=\sqrt{\bar\alpha_k}\mathbf{x}_0+\sqrt{1-\bar\alpha_k}\boldsymbol\epsilon$ follows from composing the one-step Gaussian transitions.
6. **Flow matching.** For data concentrated on two points $\pm m$ and $\mathbf{x}_0\sim\mathcal{N}(0,1)$ in 1-D, compute $v^\ast(x,\tau)$ exactly. Show that at $\tau=0$ it equals 0 (the data mean) and that as $\tau\to1$ it points toward the nearer mode.
7. **Latency.** A flow VLA has a 50 ms backbone and a 4 ms action expert. Chunks of 50 actions at 50 Hz, executing 25 per chunk. What step count $N$ keeps synchronous execution pause-free? With asynchronous execution, how stale is the observation behind each new chunk?
8. **Research question.** Two drones must pass a pylon on opposite sides to avoid each other. Each runs its own flow-matching policy with independent noise. What is the probability they choose the same side? Propose a way to correlate their choices without a central planner.

<!-- quiz: ch06 -->

## Reading assignments

1. [Ho, Jain & Abbeel (2020), DDPM](../../papers.md#ho2020) — Section 2 and Algorithms 1–2.
2. [Song, Meng & Ermon (2021), DDIM](../../papers.md#song2021) — Section 4.1.
3. [Lipman et al. (2023), Flow Matching](../../papers.md#lipman2023) — Sections 2–4; with [Liu et al. (2023), Rectified Flow](../../papers.md#liu2023flow), Section 2.
4. [Kingma & Welling (2014), Auto-Encoding Variational Bayes](../../papers.md#kingma2014) — Sections 2.2–2.4.
5. [Zhao et al. (2023), ACT](../../papers.md#zhao2023) — Section IV (the CVAE and temporal ensembling).
6. [Chi et al. (2023), Diffusion Policy](../../papers.md#chi2023) — Sections III–IV.
