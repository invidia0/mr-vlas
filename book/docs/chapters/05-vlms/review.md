# Chapter 5 Review

## Key ideas

1. A **VLM** is a language model whose context contains **image tokens**: $[\text{template};H_v;x;y]$ with $H_v=\operatorname{conn}(g_\phi(I))$. The LM is used unmodified; the image costs $N_v$ positions of prefill on every forward pass.
2. **Training stages:** (1) *alignment* — train the connector on captions with encoder and LM frozen; (2) *visual instruction tuning* — train connector and LM on instruction data with the loss only on responses. Modern recipes often train everything, including the encoder at a low learning rate. Stage 1 relies on a large LM's generality; with a tiny frozen LM it barely works (Lab 08).
3. Skills are learned in an order. In Lab 08, counting and colour come first, then relations; *where* comes last. Spatial information can be present in patch features long before the LM learns to use it. 2-D positions for image tokens help.
4. **PaliGemma**: SigLIP-So400m + Gemma-2B, prefix-LM attention (the image can read the prompt), location and segmentation tokens, all parameters trained. **SmolVLM**: pixel shuffle to 64 tokens per 512-px view, image splitting, small LMs. π0 and SmolVLA inherit these choices.
5. VLMs learn to **point and detect** from coordinates-as-tokens data. They **hallucinate** when the language prior outweighs weak visual evidence — and they will always produce *some* answer unless trained to abstain.
6. A VLM can drive a robot at five levels — **perception, task planner, code, spatial outputs, actions** — trading generality and rate against data needs and safety exposure. The last level is the VLA.

## Exercises

1. **Shapes and cost.** For PaliGemma at 448 px with a 30-token prompt and a 10-token answer, write every tensor shape from image to logits, and estimate prefill FLOPs of the LM ($\approx2N$ per token, $N\approx2.5$ B) with and without the image.
2. **Prefix-LM mask.** Write the mask for `[img ×4][prompt ×3][answer ×3]` under PaliGemma's attention and under LLaVA's causal attention. Which entries differ, and what can the image tokens "know" in each case?
3. **Stage 1 with a big LM.** Argue why projector-only alignment works with a frozen 7 B LM but poorly with Lab 08's 1 M-parameter LM. What property of large LMs makes "foreign" input tokens readable? Design an experiment that measures it.
4. **Token budgets.** A depot camera produces 1280×960 images. Compute SmolVLM's image tokens with splitting on and off. Then do the same for a 384×288 crop around a detector box. Which do you send to the VLM, and when?
5. **Hallucination.** Propose a training-data change and a decoding-time check that would make the tiny VLM of Lab 08 answer "there is no cone". How would you measure the false-negative cost (refusing when the cone *is* there)?
6. **Ladder.** For each decision in the running example — searching the yard, recognising the toolbox, grasping it, choosing which robot carries it — choose a rung of [5.5](to-action.md) and justify it by rate, data and safety.
7. **Research question.** Two drones each run a small VLM and exchange *text* descriptions of what they see. A third robot plans from those messages. Which failure modes of Chapters 4–5 propagate through the text channel, and which are filtered out? How would you test it with Lab 08's tools?

<!-- quiz: ch05 -->

## Reading assignments

1. [Liu et al. (2023), LLaVA, and LLaVA-1.5](../../papers.md#liu2023) — LLaVA Section 4; LLaVA-1.5 Section 3 (MLP projector, data mixture, response-format prompts).
2. [Beyer et al. (2024), PaliGemma](../../papers.md#beyer2024) — Section 3 and the ablations on freezing and prefix-LM attention.
3. [Marafioti et al. (2025), SmolVLM](../../papers.md#marafioti2025) — the token-budget findings.
4. [Ahn et al. (2022), SayCan](../../papers.md#ahn2022) — Sections 3–4.
5. [Liang et al. (2023), Code as Policies](../../papers.md#liang2023) — Section 3 and the examples.
