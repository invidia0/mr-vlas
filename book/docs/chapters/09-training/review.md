# Chapter 9 Review

## Key ideas

1. A VLA is trained in **stages** from abundant, general data to scarce, specific data: unimodal pretraining → VLM → robot pretraining on multi-robot mixtures → post-training on the target robot → optional RL improvement. **Co-training** with earlier data limits catastrophic forgetting; mixture weights are the key hyperparameters.
2. **Demonstrations** come from teleoperation (leader–follower, VR, handheld), scripted or privileged experts, simulation and human video. Coverage, consistency and recoveries matter more than count; record commanded actions, timestamps, calibration and outcomes.
3. **Datasets**: OXE pools ~60 datasets from 22 robots; DROID emphasises scene diversity; LIBERO is the standard simulated benchmark; the **LeRobot format** stores Parquet tables, MP4 videos and metadata including fps and normalisation statistics.
4. **LoRA** trains a zero-initialised low-rank update $\tfrac{\alpha}{r}AB$ next to frozen weights: ~1% of the parameters and a fraction of the optimiser memory; QLoRA adds a 4-bit base. Adapt action normalisation and the observation pipeline before tuning anything.
5. **Evaluation** is closed-loop and binary: report success with Wilson intervals, use paired initial conditions, evaluate per generalisation axis, filter with simulation and judge on hardware.
6. **Diversity beats volume**, and data distributions that allow **shortcuts** (scene → action, proprioception copycat) produce policies that ignore what matters.

## Exercises

1. **Memory.** Compute the training memory of full fine-tuning, LoRA ($r=16$ on all attention and MLP matrices) and QLoRA for a 3 B VLA with $d=2048$, $L=18$, SwiGLU $d_\text{ff}=16384$. Which fits on a 24 GB GPU with batch size 8 and 800-token sequences (estimate activations with gradient checkpointing)?
2. **LoRA algebra.** Show that the rank of $\Delta W=AB$ is at most $r$, and that with $B=0$ the gradient with respect to $A$ is zero at initialisation. Why does training still move away from zero?
3. **Mixture design.** You have 300 depot-arm demonstrations, access to OXE and a VQA dataset. Propose mixture weights for post-training and an experiment to tune them with at most 200 evaluation episodes.
4. **Evaluation power.** Two policies have true success rates 0.6 and 0.75. Roughly how many paired episodes do you need to detect the difference with 80% power at 5% significance? (Use a normal approximation for the difference of discordant pairs, assuming independence.)
5. **Shortcut audit.** List three shortcuts that a drone surveillance dataset could contain (think about time of day, altitude, operator, mission type), how each would show up at test time, and how you would break it in the data.
6. **Research question.** A fleet of five depot robots collects data continuously. Design a data-collection policy (which robot does which tasks where, when to ask for human interventions) that maximises *diversity* per hour rather than volume, and a way to measure it.

<!-- quiz: ch09 -->

## Reading assignments

1. [Open X-Embodiment (2024)](../../papers.md#oxe2024) — the dataset analysis and the positive-transfer results.
2. [Kim et al. (2024), OpenVLA](../../papers.md#kim2024) — Section 5 (fine-tuning experiments, LoRA vs full).
3. Hu et al. (2022), *LoRA: Low-Rank Adaptation of Large Language Models*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — Sections 4–7.
4. Khazatsky et al. (2024), *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*, [arXiv:2403.12945](https://arxiv.org/abs/2403.12945) — Sections 3–5 (collection protocol and diversity).
5. [Brohan et al. (2023), RT-2](../../papers.md#brohan2023) — the co-fine-tuning ablations.
