# Chapter 8 Review

## Key ideas

1. A **VLA** is $\pi_\theta(\mathbf{A}_t\mid\mathbf{o}_t,\ell)=h_\psi\circ f_\phi$: a pretrained vision-language backbone and an action head, trained mostly by behavioural cloning. The families are **token-based** (the backbone decodes action tokens), **action expert** (a small flow/diffusion network reads the backbone), **readout + diffusion head**, and **dual system** (slow VLM, fast action model).
2. **Inputs**: images dominate the token budget; camera identity and placement must be consistent; proprioception helps but invites copycat shortcuts; language generalises only as far as the pretrained model carries it; history is expensive.
3. **Action spaces** are the contract with the controller below: absolute vs delta, joint vs end-effector, normalisation statistics (part of the model), rotations, embodiment padding. Drones get velocity or position setpoints, never motor commands.
4. **Token heads** pay one backbone decode pass per token. FAST compresses chunks with the DCT and BPE; parallel decoding with regression (OFT) trades multimodality for speed.
5. **Action experts** run the backbone once and a small network $N$ times; they read the backbone by shared attention with a block mask (π0), by cross-attention (GR00T N1, SmolVLA), or through readout tokens (Octo).
6. **Chunking** trades reactivity for throughput and consistency; receding horizon reacts but can dither; temporal ensembling smooths but averages disagreeing modes.
7. The **closed loop** decides usefulness: train/execute at the same rate, budget the whole pipeline (not just the model), and put a safety filter and watchdog between the VLA and the actuators.

## Exercises

1. **Shapes.** For π0 with three 224-px cameras, a 20-token instruction, an 18-D padded state and 50-step chunks of 18-D actions, write the shape of every tensor from images to actions, including the block-wise attention mask.
2. **Latency.** OpenVLA (7B, bf16) on a 1 TB/s GPU: estimate the time per action. Repeat with FAST compressing a 50-step chunk into 40 tokens, and with OFT's parallel decoding. Which meets a 25 Hz arm's needs with $H_e=10$?
3. **Normalisation bug.** A policy was trained with quantile normalisation over a mixture of datasets but is deployed with statistics recomputed on one dataset. Derive the resulting error in the commanded action as a function of the two sets of statistics.
4. **Delta drift.** A delta-action policy has a per-step bias of 0.5 mm. With $H_e=20$ open-loop steps between observations, what is the maximum drift before correction? How does chunk length interact with the choice of delta vs absolute actions?
5. **Ensembling and modes.** Two consecutive chunks from a flow policy chose opposite detours around an obstacle. Compute the executed action under temporal ensembling with $m=0.1$ at the first overlapping step. Propose a rule that avoids averaging incompatible chunks.
6. **Expert coupling.** Compare the FLOPs of one denoising step for π0-style shared attention (expert width 1024, 18 layers, 50 action tokens attending to 800 cached prefix tokens) and a cross-attention expert of 6 layers reading the backbone's final 800 features.
7. **Research question.** Design a VLA for the depot *team*: one backbone shared by the arm and two drones, with embodiment-specific action heads. Which inputs does each robot need, which action space does each head output, and at which rate does each loop run?

<!-- quiz: ch08 -->

## Reading assignments

1. [Brohan et al. (2023), RT-2](../../papers.md#brohan2023) — Sections 3–4 (actions as tokens, co-fine-tuning).
2. [Kim et al. (2024), OpenVLA](../../papers.md#kim2024) — Sections 3.1–3.4 (architecture, action discretisation, design decisions).
3. [Black et al. (2024), π0](../../papers.md#black2024) — Section IV (the action expert, the attention mask, flow matching).
4. [Pertsch et al. (2025), FAST](../../papers.md#pertsch2025) — Section 4 (the tokenizer).
5. [Zhao et al. (2023), ACT](../../papers.md#zhao2023) — temporal ensembling and chunk-size ablations.
