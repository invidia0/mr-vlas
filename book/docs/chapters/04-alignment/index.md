# 4. Multimodal Representation and Alignment

Chapter 3 produced visual features; Chapter 2 produced language models. This chapter connects them. It covers the contrastive objective that places images and text in one shared space, the two models that made it work at scale (CLIP and SigLIP), and the connectors that feed visual tokens into a language model. It then turns to grounding, which turns a phrase into an image region, and to the systematic ways aligned representations fail — failures every VLA built on them inherits.

By the end you should be able to derive and implement InfoNCE and the sigmoid loss, explain what their gradients do, choose a connector for a token budget, write a bounding box as tokens, and design a probe that tests whether an encoder can be trusted with a given instruction.

## Learning objectives

1. Derive the InfoNCE loss, its gradient (pull the positive, push weighted negatives), the role of temperature and batch size, and its mutual-information bound. ([4.1](contrastive.md))
2. Explain CLIP's training and zero-shot classification, the effect of prompts, and why SigLIP's sigmoid loss scales better. ([4.2](clip-siglip.md))
3. Compare projectors, pixel shuffle, learned-query resamplers and gated cross-attention by tokens, cost and what they lose. ([4.3](connectors.md))
4. Describe four families of visual grounding and the coordinates-as-tokens idea that VLAs reuse for actions. ([4.4](grounding.md))
5. Explain the modality gap, bag-of-words behaviour, spatial blindness and typographic attacks from the training objective, and measure them. ([4.5](failure-modes.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [4.1](contrastive.md) | Contrastive learning and shared embedding spaces | Lab 07 |
| [4.2](clip-siglip.md) | CLIP and SigLIP | Lab 07 |
| [4.3](connectors.md) | Connectors: projectors, resamplers, gated cross-attention | Lab 08 |
| [4.4](grounding.md) | Grounding: from words to image regions | Lab 07 (exercise) |
| [4.5](failure-modes.md) | Failure modes of aligned representations | Lab 07 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 07 — Contrastive alignment** (`labs/07_contrastive/`): InfoNCE by hand; a tiny CLIP trained on depot scenes and captions (retrieval, zero-shot classification, binding probes); the sigmoid loss; CLIP ViT-B/32 on the same probes, prompt sensitivity, and the modality gap.
