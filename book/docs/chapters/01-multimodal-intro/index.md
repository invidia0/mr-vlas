# 1. From Vision and Language Models to Multimodal Models

Chapter 0 introduced the building blocks. This chapter looks at the two modalities a VLA reads — **language** and **images** — as *data and models* in their own right, then asks the question the rest of the book keeps answering in more detail: **how do you combine information from different modalities in one network?**

The chapter is intentionally architectural rather than mathematical. The heavy mathematics of transformers comes in [Chapter 2](../02-transformers/index.md), of vision encoders in [Chapter 3](../03-vision/index.md), of alignment in [Chapter 4](../04-alignment/index.md). Here the goal is to fix the *design space*: what each modality looks like as a tensor, what a model of it produces, and which ways of fusing them exist.

## Learning objectives

1. Describe how a language model turns a prompt into text: logits, decoding strategies, context windows, chat templates. ([1.1](language.md))
2. Describe an image as a tensor, explain why raw pixels are a poor representation, and what a pretrained visual feature is. ([1.2](vision.md))
3. Enumerate the main fusion strategies — early, late, and intermediate fusion; concatenation, cross-attention, FiLM, adaptive normalisation — and their costs. ([1.3](combining.md))
4. Explain the **symbol grounding problem** and why it is the reason VLAs need robot data at all. ([1.3](combining.md#the-grounding-problem))

## Sections

| Section | Topic |
|---|---|
| [1.1](language.md) | Language models in operation |
| [1.2](vision.md) | Images as data |
| [1.3](combining.md) | Combining modalities |
| [Review](review.md) | Quiz and reading assignments |

## Labs

This chapter reuses Labs 01–02 (tokenization and embeddings) and previews Lab 05 (inspecting a pretrained language model), where you run the decoding strategies of [1.1](language.md#decoding) on a real model.
