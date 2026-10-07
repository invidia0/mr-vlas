# Chapter 4 Review

## Key ideas

1. **Contrastive learning** places matching image–text pairs close together on a unit sphere. **InfoNCE** is a $B$-way classification per row and per column of the $B\times B$ similarity matrix; its gradient pulls the positive and pushes negatives in proportion to their softmax weight, so **hard negatives** dominate. The temperature sets how peaked that weighting is; $\log B$ bounds the mutual information it can certify.
2. **CLIP** trained two towers on 400 M web pairs and made zero-shot classification by text prompts work. Prompts matter (69% → 95% on our renders). **SigLIP** replaces the softmax by an independent sigmoid per pair, with a learned bias initialised at −10; no batch-wide normaliser means cheap large batches and good small-batch training.
3. **Connectors** translate patch features into an LLM's token space and set the image-token budget: projectors (one token per patch), pixel shuffle (fewer, wider tokens), resamplers (a fixed number of learned queries), Flamingo's gated cross-attention (images read inside the LM, gates initialised at zero).
4. **Grounding** maps phrases to regions: dense similarity maps, open-vocabulary detectors, coordinates as tokens (PaliGemma's 1024 location tokens), promptable segmentation. Discretising a continuous quantity into tokens is the idea VLAs reuse for actions.
5. Aligned representations fail systematically: the **modality gap**, **bag-of-words** behaviour, **spatial and numerical blindness**, **typographic** and spurious cues, **prompt and distribution sensitivity** — all predictable from "the loss only rewards what distinguishes a caption from the other captions in the batch".

## Exercises

1. **InfoNCE gradient.** Derive $\partial\mathcal{L}_{I\to T}/\partial\mathbf{v}_j$ for a negative caption $j\ne i$. Show that it vanishes as $p_{ij}\to0$, and compute it when all embeddings are identical.
2. **Temperature extremes.** With all embeddings normalised, what is the maximum and minimum value of InfoNCE for $B=256$ at $\tau=1$? Why can a model with $\tau=1$ never reach low loss, however good its embeddings?
3. **SigLIP initialisation.** At initialisation, cosines are near 0. Compute the initial SigLIP loss for $B=256$ with $t=10$, $b=-10$, and with $b=0$. Which terms dominate in each case, and why does the bias help?
4. **Token budgets.** A VLM uses SigLIP-So400m at 384 px ($P=14$) and a Gemma-2B LM ($d=2048$, $L=18$). Compare the LM prefill FLOPs for one image with (a) an MLP projector, (b) pixel shuffle $r=3$ (cropping the grid as needed), (c) a 64-query resampler.
5. **Location tokens.** A drone image is 1920×1080 and the VLM normalises both axes to 1024 bins. What is the worst-case quantisation error in pixels along each axis? At 30 m altitude with a 90° horizontal field of view, what is it in metres?
6. **Probe design.** Design a probe set (images and captions) that tests whether an encoder understands "nearest to the drone" in top-down depot images. What are the hard negatives?
7. **Research question.** Two drones see the same yard from different altitudes and share embeddings rather than images. Which failure modes of this chapter would corrupt a contrastive *place-recognition* embedding trained on single-drone data, and how would you construct positives and negatives from multi-drone logs to fix them?

<!-- quiz: ch04 -->

## Reading assignments

1. [van den Oord et al. (2018), CPC](../../papers.md#oord2018) — Section 2.3 (the InfoNCE loss and the mutual-information bound).
2. [Radford et al. (2021), CLIP](../../papers.md#radford2021) — Figure 1, Sections 2.3–2.5, Figure 2 (efficiency of contrastive vs predictive objectives), Section 3.1.4 (prompt engineering).
3. [Zhai et al. (2023), SigLIP](../../papers.md#zhai2023) — Section 3 (the loss and its efficient implementation) and Figure 2.
4. [Alayrac et al. (2022), Flamingo](../../papers.md#alayrac2022) — Figure 3 and Section 2.1 (Perceiver resampler, gated cross-attention).
5. [Li et al. (2023), BLIP-2](../../papers.md#li2023) — Figure 2 (the Q-Former and its three objectives).
6. [Yuksekgonul et al. (2023), When and why VLMs behave like bags-of-words](../../papers.md#yuksekgonul2023) — Sections 2–4.
