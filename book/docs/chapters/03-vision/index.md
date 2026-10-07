# 3. Vision Architectures

Part II turns to the half of a VLA that sees. This chapter builds the networks that turn camera images into feature vectors: convolutional networks, as history and as the clearest case study of inductive bias; the Vision Transformer, which every VLA backbone uses; the self-supervised encoders whose features know where objects and their parts are; and the engineering of choosing an encoder, a resolution and a token budget for a robot.

By the end you should be able to write a ViT from memory, state the shape of every tensor in it, estimate its cost at any resolution, and argue which encoder a given robot should use.

## Learning objectives

1. Define a convolution layer precisely, derive its output size, parameter count and FLOPs, and explain locality, weight sharing and translation equivariance — and measure them. ([3.1](cnn.md))
2. Build the Vision Transformer, count its parameters and FLOPs, explain the role of the [CLS] token, position embeddings and their interpolation, and register tokens. ([3.2](vit.md))
3. Explain how DINO avoids representation collapse (EMA teacher, centering, sharpening, multi-crop), what DINOv2 added, and how MAE learns by reconstruction. ([3.3](self-supervised.md))
4. Choose an encoder, a resolution and a token-reduction method for a robot, and estimate whether a given object is visible at all. ([3.4](encoders-for-vlas.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [3.1](cnn.md) | Convolutional networks | Lab 06 |
| [3.2](vit.md) | The Vision Transformer | Lab 06 |
| [3.3](self-supervised.md) | Self-supervised encoders: DINO, DINOv2, MAE | Lab 06 |
| [3.4](encoders-for-vlas.md) | What a VLA needs from its vision encoder | Labs 06, 08 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 06 — Vision encoders** (`labs/06_vision_encoders/`): convolution by hand, equivariance and receptive fields measured, CNN vs ViT under scarce data and unseen positions, DINOv2-S opened up (patch-feature PCA, attention, an artifact token, resolution cost).

The chapter also introduces `vlacourse.data.scenes`, the procedural generator of depot scenes (toolboxes, crates, barrels and cones in four colours, seen from above) that every model from here to the toy VLA is trained on.
