# 10. VLA Architectures in Practice

This chapter reads the systems that defined the field through the lens of Chapters 2–9. For each one it answers the same ten questions: architecture, inputs, outputs, training objective, data, inference, compute, strengths, limitations, and what was genuinely new. The point is not to memorise model cards. It is to be able to place any new VLA paper in the design space, predict its cost, and see which of its ideas you could reuse.

## The design space in one table

| System | Year | Backbone | Action head | Chunk | Size | Data | Distinctive idea |
|---|---|---|---|---|---|---|---|
| [ACT](precursors.md) | 2023 | ResNet-18 + transformer | CVAE decoder | 100 | 80 M | ~50 demos / task | action chunking, temporal ensembling |
| [Diffusion Policy](precursors.md) | 2023 | ResNet-18 + U-Net / transformer | diffusion | 16 (exec. 8) | tens of M | tens–hundreds / task | generative action heads |
| [RT-1](rt.md) | 2022 | EfficientNet + FiLM + transformer | 256-bin tokens | 1 | 35 M | 130 k episodes | scale and diversity of real data |
| [RT-2](rt.md) | 2023 | PaLI-X / PaLM-E VLM | text tokens | 1 | 5–55 B | RT-1 data + web | web knowledge → actions; co-fine-tuning |
| [Octo](octo-openvla.md) | 2024 | transformer from scratch | diffusion on readouts | few | 27–93 M | 800 k OXE episodes | adaptable generalist |
| [OpenVLA](octo-openvla.md) | 2024 | SigLIP+DINOv2 + Llama 2 | 7 action tokens | 1 | 7.5 B | 970 k OXE episodes | open token-based VLA, design study |
| [OpenVLA-OFT](octo-openvla.md) | 2025 | same | parallel L1 regression | chunk | 7.5 B | fine-tuning | fast fine-tuned decoding |
| [π0](pi0.md) | 2024 | PaliGemma 3 B | flow-matching expert (shared attention) | 50 | 3.3 B | 10 k+ hours, 7 robots | expert + long chunks at 50 Hz |
| [π0-FAST](pi0.md) | 2025 | PaliGemma 3 B | FAST tokens | 50 (compressed) | 3 B | same | compressed action tokens |
| [π0.5](pi0.md) | 2025 | π0 design | subtask text + flow expert | 50 | ~3 B | heterogeneous co-training | open-world generalisation |
| [GR00T N1](groot-smolvla.md) | 2025 | Eagle-2 VLM | DiT flow head (System 1) | chunk | ~2 B | data pyramid incl. video, sim | dual system for humanoids |
| [SmolVLA](groot-smolvla.md) | 2025 | SmolVLM2 (half the layers) | flow expert (cross/self-attention) | 50 | 0.45 B | community LeRobot data | efficiency, async inference |

## Sections

| Section | Systems | Lab |
|---|---|---|
| [10.1](precursors.md) | ACT, Diffusion Policy | Labs 09, 11 |
| [10.2](rt.md) | RT-1, RT-2 | Lab 11 |
| [10.3](octo-openvla.md) | Octo, OpenVLA, OpenVLA-OFT | Labs 11, 13 |
| [10.4](pi0.md) | π0, FAST, π0.5 | Lab 11 |
| [10.5](groot-smolvla.md) | GR00T N1, SmolVLA | Lab 14 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 14 — Inside SmolVLA** (`labs/14_smolvla/`): load SmolVLA with the `lerobot` library, feed it a sample from a LeRobot dataset, inspect every input and output tensor, count tokens per camera, time the backbone and the flow expert separately, and check the latency model of [6.6](../06-generative/sampling-cost.md) against measurements.
