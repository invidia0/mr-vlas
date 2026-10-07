# Chapter 10 Review

## Key ideas

1. **ACT** and **Diffusion Policy** contributed action chunks, temporal ensembling and generative action heads — still the strongest baselines for narrow tasks.
2. **RT-1** showed what scale and diversity of real-robot data buy; **RT-2** showed that a VLM writing action tokens transfers web knowledge to new semantic situations, but not new motions.
3. **Octo** (small, from scratch, diffusion head, adaptable) and **OpenVLA** (7 B VLM, action tokens) bracket the open generalists; **OpenVLA-OFT** shows how much fine-tuning recipes (parallel decoding, chunking, continuous outputs) change speed and success.
4. **π0** established the VLM + flow-matching expert + long-chunk template; **FAST** made token-based training viable at high rates; **π0.5** showed that heterogeneous co-training drives open-world generalisation.
5. **GR00T N1** separates slow semantics and fast motor generation and uses a data pyramid; **SmolVLA** shows how far efficiency choices (64 tokens, half the layers, small expert, asynchronous execution) can take a VLA on modest hardware.

## Exercises

1. **Read a new paper.** Take any VLA paper not in this chapter. Fill in the ten-question template, draw the token sequence with shapes, and estimate its per-chunk latency on an embedded GPU with the formulas of [6.6](../06-generative/sampling-cost.md) and [8.7](../08-vla/closed-loop.md).
2. **RT-2's limits.** RT-2 generalised to "move the banana to the number three" but not to motions absent from its robot data. Explain this with the training objective and the data, and propose a data source that could teach new motions.
3. **OpenVLA vs OFT.** Compute the per-action latency of OpenVLA (7 sequential tokens) and of OFT (one parallel pass producing a chunk of 8 actions) for a 7 B bf16 model on a 200 GB/s device. What is the speed-up per action?
4. **π0's mask.** Write π0's attention mask for 2 cameras, a 10-token instruction, 1 state token and 4 action tokens, and identify which keys/values can be cached across denoising steps.
5. **Choosing a starting point.** For each of the following, choose a system from this chapter and justify it: (a) a single precise insertion task with 50 demonstrations; (b) a household robot that must follow open-ended instructions; (c) a fleet of low-cost arms with a shared GPU; (d) a drone that must recognise and approach objects named in natural language.
6. **Research question.** Design "SmolVLA for teams": a shared backbone serving several robots, with per-robot adapters and action experts. Where would messages from other robots enter the token sequence, and what would that cost per robot per step?

<!-- quiz: ch10 -->

## Reading assignments

1. [Zhao et al. (2023), ACT](../../papers.md#zhao2023) and [Chi et al. (2023), Diffusion Policy](../../papers.md#chi2023) — the ablation sections.
2. [Brohan et al. (2023), RT-2](../../papers.md#brohan2023) — Section 4 and the emergent-capability experiments.
3. [Kim et al. (2024), OpenVLA](../../papers.md#kim2024) — Section 3.4 (design decisions).
4. [Black et al. (2024), π0](../../papers.md#black2024) — Section IV; then π0.5's data-ablation results.
5. [Shukor et al. (2025), SmolVLA](../../papers.md#shukor2025) — the efficiency design and asynchronous inference.
6. [NVIDIA (2025), GR00T N1](../../papers.md#gr00t2025) — the dual-system architecture and data pyramid.
