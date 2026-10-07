# 5. Vision-Language Models

This chapter assembles everything in Part II into the model that every token-based VLA starts from: a **vision-language model**. A pretrained vision encoder, a connector and a pretrained language model read image tokens and text tokens as one sequence. We build one from parts in Lab 08, train it in the classic stages, and find out what it cannot do and why. Then we dissect the small VLMs that VLAs are actually built on, and end Part II by asking how a model that *describes* can be made to *act*.

By the end you should be able to trace every tensor of a VLM from pixels to logits, explain what each training stage changes, predict a VLM's token budget and latency, diagnose spatial and hallucination failures, and place a VLM at the right level of a robot's decision hierarchy.

## Learning objectives

1. Describe the LLaVA-style architecture and its sequence layout, and compute the shapes and image-token cost end to end. ([5.1](recipe.md))
2. Explain the alignment and instruction-tuning stages: what is frozen, what data is used, which loss mask applies, and why a small frozen LM aligns poorly. ([5.2](training.md))
3. Compare PaliGemma and SmolVLM: prefix-LM attention, location tokens, pixel shuffle, image splitting, and their consequences for VLAs. ([5.3](small-vlms.md))
4. Explain how VLMs learn to point, why spatial questions are hard, and why VLMs hallucinate — and measure all three. ([5.4](spatial.md))
5. Place a VLM on the ladder from perception module to VLA, and state the trade-offs of each rung. ([5.5](to-action.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [5.1](recipe.md) | The vision-language model recipe | Lab 08 |
| [5.2](training.md) | Training VLMs: alignment and visual instruction tuning | Lab 08 |
| [5.3](small-vlms.md) | Inside modern small VLMs: PaliGemma and SmolVLM | Lab 08 |
| [5.4](spatial.md) | Grounding, spatial reasoning and hallucination | Lab 08 |
| [5.5](to-action.md) | From describing to acting | |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 08 — Build a VLM, then open a real one** (`labs/08_vlm/`): a tiny LLaVA-style VLM from the Lab 07 image tower, an MLP projector and TinyGPT; alignment and instruction-tuning stages; per-skill learning curves; 1-D vs 2-D positions for image tokens; attention maps; hallucination; SmolVLM-256M dissected (pixel shuffle, image splitting, spatial questions).
