# Vision-Language-Action Models: From First Principles to Multi-Robot Systems

An interactive, research-level textbook and lab course on **Vision-Language-Action (VLA) models** —
neural networks that read camera images and a natural-language instruction and output robot actions —
built from the definition of a tensor up to open research problems in **UAVs** and **multi-robot systems**.

The course does not teach how to call a VLA API. It teaches what happens underneath — tokens, embeddings,
attention, transformers, vision encoders, multimodal alignment, generative models, imitation and
reinforcement learning — so that you can read, implement, debug, modify and do research on VLA systems,
and integrate them *safely* into real robot stacks (ROS 2, PX4).

## Learning objectives

After the course you should be able to:

1. Derive and implement every component of a modern VLA (tokenizer, embeddings, transformer, ViT,
   projector, action head) and state the shape of every tensor in it.
2. Explain *why* each architectural choice was made, and its cost in memory, compute and latency.
3. Train toy policies by behavioural cloning, with autoregressive, diffusion and flow-matching action
   heads, and diagnose their failure modes (multimodality, compounding error, distribution shift).
4. Dissect published VLAs (RT-2, Octo, OpenVLA, π0, GR00T N1, SmolVLA) mechanistically.
5. Design a deployment architecture that keeps safety-critical control outside the learned model,
   in particular for PX4-based UAVs.
6. Formulate research questions on VLAs for multi-robot systems and aerial robots.

## Prerequisites

Linear algebra, calculus, probability, optimisation; Python and basic PyTorch; robotics and control
fundamentals. **No** prior knowledge of deep learning, LLMs, transformers, diffusion models or
reinforcement learning is assumed.

## Course outline

All five parts are written, with quizzes and executed labs.

| Part | Chapters | Labs |
|---|---|---|
| I — Foundations | 0 Foundations · 1 Vision, language, multimodal · 2 Transformers in depth | 00–05 |
| II — Seeing and reading | 3 Vision architectures · 4 Multimodal alignment · 5 VLMs | 06–08 |
| III — Acting | 6 Generative models · 7 Imitation & RL · 8 VLAs · 9 Training VLAs · 10 Case studies | 09–13 |
| IV — Deploying | 11 Deployment · 12 Embodied AI · 13 UAVs (PX4 safety architecture) | 14–15 |
| V — Research | 14 Multi-robot systems · 15 Research directions | 16 |

## Repository structure

```
.
├── README.md, LICENSE
├── pyproject.toml, uv.lock, .python-version   reproducible Python 3.12 environment (uv)
├── mkdocs.yml                                 book configuration (MkDocs Material)
├── book/
│   ├── docs/                                  the book's Markdown sources
│   │   ├── index.md                           introduction and running example
│   │   ├── chapters/NN-slug/                  one directory per chapter (sections + review page)
│   │   ├── appendix/                          MuJoCo guide, ROS 2 + PX4, notation
│   │   ├── glossary.md, papers.md             glossary and curated reading list
│   │   └── assets/                            quiz.js, KaTeX loader, CSS, figures/ (generated SVGs)
│   └── hooks/                                 quiz.py: renders quizzes/*.yml; figures.py: dark-theme figures
├── quizzes/chNN.yml                           interactive quiz questions with per-option explanations
├── labs/NN_slug/
│   ├── NN_slug.py                             lab source (jupytext "percent" format; diff-friendly)
│   └── NN_slug.ipynb                          the notebook, executed, with expected outputs
├── src/vlacourse/                             reference implementations the labs build and then reuse
│   ├── attention.py, transformer.py           attention, MHA with KV cache, TinyGPT
│   ├── data/instructions.py                   procedural "depot" instruction corpus
│   ├── data/scenes.py                         procedural top-down depot scenes: images, captions, Q&A
│   ├── vision.py, clip.py, vlm.py             ViT, tiny CLIP (InfoNCE / SigLIP losses), tiny VLM
│   ├── generative.py                          flow matching and action binning (Lab 09)
│   ├── depot_env.py                           2-D depot navigation task, scripted expert, collect, evaluate
│   ├── vla.py                                 TinyVLA: regression / action-token / flow-matching heads
│   ├── mujoco_depot.py                        the depot in MuJoCo physics (Lab 12, Appendix A)
│   ├── lora.py                                low-rank adaptation (Lab 13)
│   ├── drone_depot.py                         the depot as a lagged, wind-blown drone (Lab 15)
│   ├── safety.py                              CBF safety filter and watchdog (Labs 15–16)
│   ├── teams.py                               auction, greedy, lossy channel, decentralised auction (Lab 16)
│   └── seed.py                                seeding and device helpers
├── ros2/vla_px4_bridge/                       ROS 2 package: safety supervisor between VLA goals and PX4 (App. B)
├── tests/                                     pytest checks of src/ (causality, KV cache, init, ...)
├── scripts/
│   ├── build_labs.sh                          .py -> .ipynb and execute (regenerates expected outputs)
│   ├── make_figures.py                        draws the book's figures (light + dark SVG)
│   ├── setup_lerobot_env.sh                   separate environment for Lab 14 (lerobot + SmolVLA)
│   ├── patch_md.py                            apply narrative fixes to a lab source and its executed notebook
│   └── nb_text.py                             print a notebook's text outputs
└── data/                                      downloads and model cache (git-ignored)
```

## Installation

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 itself if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is not installed
git clone <this repository> vla-course && cd vla-course
uv sync                                             # creates .venv/ with all dependencies (incl. docs + dev tools)
uv run pytest                                       # sanity check: 21 tests should pass (≈2 min)
```

`uv sync` installs exactly the versions in `uv.lock`. The first sync downloads ≈3 GB (PyTorch with CUDA
libraries).

### GPU, CUDA and CPU-only machines

| | |
|---|---|
| PyTorch | 2.11, wheels built for **CUDA 12.8** (`pytorch-cu128` index in `pyproject.toml`) |
| NVIDIA driver | ≥ 570 (any driver supporting CUDA 12.8; Blackwell GPUs require cu128 wheels) |
| CPU-only | works with the same wheels; no change needed. All Part I labs run on CPU. |
| macOS | PyTorch comes from PyPI (CPU/MPS) automatically. |

Check your setup:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Reading the book

```bash
uv run mkdocs serve          # then open http://127.0.0.1:8000
```

or build a static site (`site/`) that can be opened from disk:

```bash
uv run mkdocs build          # strict mode: fails on any broken link, anchor or invalid quiz
```

The book has a navigation sidebar, full-text search, rendered equations (KaTeX), Mermaid diagrams,
interactive quizzes at the end of each chapter, cross-links between every concept and its definition, a
glossary and a curated paper list. KaTeX and Mermaid are loaded from a CDN, so an internet connection is
needed for equations and diagrams to render.

## Running the labs

```bash
uv run jupyter lab           # open labs/00_nn_from_scratch/00_nn_from_scratch.ipynb, etc.
```

Each lab starts with its goal, the book sections it implements, its runtime, hardware needs and seed. The
committed notebooks already contain the **expected outputs**; re-running them should reproduce the same
numbers up to floating-point and hardware differences (GPU timings naturally differ).

| Lab | Topic | Runtime | Hardware / downloads |
|---|---|---|---|
| 00 | Tensors, NN from scratch, backprop, gradient check, regression to the mean (planar-arm IK) | < 1 min | CPU |
| 01 | Tokenization: char, word, BPE from scratch; SmolLM2 tokenizer | < 1 min | CPU; 2 MB |
| 02 | Embeddings, skip-gram, RNN gradients, sinusoidal and rotary positions | ≈ 1 min | CPU |
| 03 | Attention from scratch: scaling, masks, equivariance, MHA, cross-attention, learned recall | < 1 min | CPU |
| 04 | Transformer block, TinyGPT trained on depot instructions, KV cache | ≈ 1–2 min | CPU or GPU |
| 05 | Inside SmolLM2-135M: params, attention heads, logit lens, decoding, prefill/decode timing | ≈ 1 min | 270 MB; GPU optional (< 1 GB) |
| 06 | Convolution by hand, equivariance, receptive fields, CNN vs ViT, DINOv2 opened up | ≈ 2 min GPU / 8 min CPU | 90 MB; GPU optional (< 1 GB) |
| 07 | InfoNCE by hand, a tiny CLIP and SigLIP on depot scenes, binding probes, CLIP ViT-B/32, modality gap | ≈ 4 min GPU / 25 min CPU | 600 MB; GPU optional (< 2 GB) |
| 08 | A tiny LLaVA-style VLM in two stages, skill curves, 2-D image positions, hallucination, SmolVLM-256M | ≈ 12 min GPU / 1.5 h CPU | 510 MB; GPU recommended (≈ 1.5 GB) |
| 09 | Mode averaging, binning, mixtures, flow matching on multimodal actions; sampling cost | ≈ 4 min | CPU |
| 10 | Behavioural cloning, compounding error, DART, DAgger, residual RL fine-tuning on a patrol task | ≈ 3 min | CPU |
| 11 | A toy VLA in the depot: the instruction-ignoring shortcut and its fix, three action heads, execution schemes, timing | ≈ 35–60 min | GPU (≈ 2 GB); needs Lab 07 |
| 12 | The depot in MuJoCo: MJCF, actuator lag, cameras, deprojection, sim-to-sim gap, fine-tuning vs co-training | ≈ 20 min | GPU (≈ 1.5 GB); needs Lab 11 |
| 13 | Recording, the LeRobot format by hand (Parquet + MP4), codec effects, LoRA vs full fine-tuning, co-training, paired tests | ≈ 30–45 min | GPU (≈ 1.5 GB); 100 MB disk |
| 14 | Inside SmolVLA: processors, normalisation, tokens, stage latency, flow steps, simulated quantisation, a policy server | ≈ 6 min | separate env; 1.5 GB downloads; GPU (≈ 1.3 GB) |
| 15 | Flying the toy VLA as a drone: direct vs goal-level control, latency, CBF margins, watchdog, link loss | ≈ 8 min | GPU (< 1 GB); needs Lab 11 |
| 16 | A team of VLAs: independent execution, centralised and decentralised auctions over a lossy channel, separation | ≈ 5 min | GPU (< 1 GB); needs Lab 11 |

Labs are authored as `labs/NN_slug/NN_slug.py` (jupytext percent format) and converted. To regenerate a
notebook after editing its source (or all of them):

```bash
scripts/build_labs.sh labs/03_attention/03_attention.py
scripts/build_labs.sh              # all labs
uv run pytest --nbmake labs/       # execute every notebook as a test
```

Model weights and datasets are cached in `data/` (`HF_HOME=data/hf`).

## How the labs are organised

The labs follow the book. Every mechanism is first implemented by hand in the notebook, with intermediate
tensors printed and plotted; the corresponding reference implementation in `src/vlacourse/` is then
checked for equality and reused by later labs, so that you never use a component you have not built.
Pretrained models (SmolLM2, DINOv2, CLIP, SmolVLM, SmolVLA) are only opened after their mechanisms have
been built from scratch.

Every lab states its runtime, memory and downloads. All models fit a 6 GB laptop GPU; Part I runs on CPU.
Later labs build on earlier checkpoints: Lab 07 → `tiny_clip.pt` (used by Labs 11–16), Lab 11 →
`toy_vla_mse.pt` (used by Labs 12, 15, 16).

**Lab 14 runs in a separate environment**, because `lerobot` pins older `numpy`/`transformers` than the course
environment: `scripts/setup_lerobot_env.sh` creates `data/venv-lerobot`, and `scripts/build_labs.sh` uses it
automatically for that lab.

**ROS 2 and PX4** are optional and not part of the Python environment. Appendix B explains how to run the
`ros2/vla_px4_bridge` package against PX4 SITL in an Ubuntu 24.04 container; that integration was written for
this course but not executed on the build machine.

## Reproducibility

- Python and package versions pinned by `.python-version` and `uv.lock`.
- Every lab and test fixes its seeds (`vlacourse.seed.set_seed`).
- Expected outputs are stored in the executed notebooks; `scripts/build_labs.sh` regenerates them.
- Book figures are generated, not drawn by hand: `uv run python scripts/make_figures.py [name ...]`.
- The book build is strict: broken internal links, missing anchors and malformed quizzes fail it.

## License and citation

Course text and code are released under the [MIT License](LICENSE) © 2026 Mattia Mantovani. Papers referenced in the book belong to their authors; links point
to official sources.
