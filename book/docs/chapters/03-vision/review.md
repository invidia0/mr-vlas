# Chapter 3 Review

## Key ideas

1. A **convolution** applies one small kernel at every position: locality and weight sharing cut parameters from $O(HW)$ to $O(k^2)$ per filter and make the layer **translation-equivariant**; global pooling then makes predictions **invariant**. Receptive fields grow linearly with depth, faster with strides.
2. The **ViT** embeds $P\times P$ patches with one shared linear map (a strided convolution), adds learned position embeddings and runs a bidirectional transformer encoder. Cost: $\approx2N_p\cdot12Ld^2+4LN_p^2d$ FLOPs; $N_p=HW/P^2$, so resolution is a token budget.
3. Weak inductive bias makes ViTs data-hungry: worse than CNNs on small data, better at scale. In VLAs they are always pretrained.
4. **[CLS]** pools the image into one vector; VLMs and VLAs keep the **patch tokens**, because a single vector loses *where*. Large ViTs grow **artifact tokens**; **registers** fix them.
5. **DINO** learns by self-distillation: an EMA teacher, centering and sharpening against collapse, multi-crop for spatially meaningful features. **DINOv2** adds curation, a patch-level masked loss and distillation. **MAE** reconstructs 75% masked patches with an encoder that sees only the visible quarter.
6. A VLA's encoder choice trades **semantics** (language-supervised), **geometry** (self-supervised), **tokens**, **robustness** and **latency**. Fusing SigLIP and DINOv2, reducing tokens (pooling, pixel shuffle, resamplers, cropping) and fine-tuning the encoder are the main levers.
7. Before choosing a resolution, compute **pixels per object** from the camera geometry.

## Exercises

1. **Output sizes.** A $224\times224\times3$ image passes through a $7\times7$ convolution with stride 2 and padding 3 (64 channels), then a $3\times3$ max pool with stride 2 and padding 1. Give each output shape, the parameter count, and the FLOPs of the convolution.
2. **Receptive field.** Compute the theoretical receptive field of the last unit of `SmallCNN` from [3.1](cnn.md#implementation). Is it larger than the 64-pixel image?
3. **ViT accounting.** For SigLIP So400m/14 ($d=1152$, $L=27$, MLP width 4304) at 224 and 448 px, compute $N_p$, the parameters per block, and the FLOPs per image. At which resolution does the attention term exceed 10% of the total?
4. **Position interpolation.** A ViT trained at 224 px with $P=16$ is run at 384 px. What is the new grid, and what must happen to $E_\text{pos}$? Why does the [CLS] row not need interpolating?
5. **Collapse.** Show that, without centering, the DINO loss is minimised by a teacher and student that output the same one-hot vector for every image. Show that without sharpening (with $\tau_t=\tau_s$ and centering) the uniform distribution is a fixed point.
6. **Token budget.** π0 reads three cameras at 256 tokens each, plus 20 instruction tokens. Replace its encoder with SmolVLA's (64 tokens per image). By what factor does the backbone's prefill compute fall (ignore the encoder itself)? What might be lost?
7. **Research question.** A team of drones shares one ground GPU. Each drone sends either (a) its full 224-px image every second, or (b) 64 pixel-shuffled SigLIP tokens computed on board, or (c) only the tile around a detector hit. Compare uplink bandwidth (bytes per second) and what each option lets the ground model do.

<!-- quiz: ch03 -->

## Reading assignments

1. [He et al. (2016), ResNet](../../papers.md#he2016) — Sections 3.1–3.2: the residual formulation. Compare with the residual stream of [2.4](../02-transformers/block.md#the-residual-stream).
2. [Dosovitskiy et al. (2021), ViT](../../papers.md#dosovitskiy2021) — Section 3.1 (equations 1–4) with [3.2](vit.md) open; then Figure 3 (data size vs CNNs) and Figure 7 (position-embedding similarity).
3. [Caron et al. (2021), DINO](../../papers.md#caron2021) — Algorithm 1 and Section 3.1; then Figure 1. Reproduce the collapse analysis of exercise 5 from Section 5.3.
4. [Oquab et al. (2024), DINOv2](../../papers.md#oquab2024) — Sections 3–4 (data curation and the training recipe).
5. [He et al. (2022), MAE](../../papers.md#he2022) — Figure 1 and Section 3; the masking-ratio ablation (Figure 5).
6. [Darcet et al. (2024), Vision Transformers Need Registers](../../papers.md#darcet2024) — Figures 1–2.
