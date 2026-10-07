# Chapter 15 Review

## Key ideas

1. **Multi-robot VLAs**: teammate-aware policies, decentralised and shared foundation models, reliable language coordination (language states the problem, solvers solve it), and team-level evaluation are open; team data is the main bottleneck.
2. **Aerial VLAs**: early systems span mission writers (UAV-VLA) and direct-command VLAs (RaceVLA, CognitiveDrone, which gains from a separate reasoning step); goal-level aerial VLAs, aerial perception, latency-aware control and safety evidence are open.
3. **Long horizons, generalisation, data efficiency**: memory and sub-goal structure, detection of shortcuts, diversity per demonstration, sim-to-real at the goal level, and adaptation from few demonstrations.
4. **Safety, uncertainty, interpretability**: calibrated uncertainty for continuous actions (conformal methods), learning from filter interventions, and probes that separate grounding errors from control errors.
5. **Real time and communication**: latency-aware training, slow/fast splits, event-triggered and bandwidth-aware messages between robots.
6. **Methodology**: comparisons with effect sizes, strong simple baselines, controlled data, paired evaluation with intervals, stress tests, honest simulation claims, reproducibility.

## Exercises

1. **Pick a question.** Choose one research question from this chapter. Write it as a comparison ([15.6](methodology.md)), name baselines that could win, the metric, the effect size that matters, and the number of paired episodes needed to detect it with 95% confidence.
2. **Design the first experiment.** For the same question, write the experiment you would run with the course tools in under a week of compute on one GPU, including what you would log.
3. **Find the trap.** For a recent VLA paper of your choice, identify which items of the checklist in [15.6](methodology.md) it satisfies and which it does not; propose the single additional experiment that would most strengthen its claim.
4. **Cross the boundaries.** Combine two sections: e.g. uncertainty-aware allocation (15.1 + 15.4), or latency-aware goal-level drones (15.2 + 15.5). State the question and why the combination is more than the sum of its parts.

<!-- quiz: ch15 -->

## Reading assignments

1. Black, Galliker & Levine (2025), *Real-Time Execution of Action Chunking Flow Policies*, [arXiv:2506.07339](https://arxiv.org/abs/2506.07339).
2. Ren et al. (2023), *Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners* (KnowNo), [arXiv:2307.01928](https://arxiv.org/abs/2307.01928).
3. Lykov et al. (2025), *CognitiveDrone*, [arXiv:2503.01378](https://arxiv.org/abs/2503.01378); Serpiva et al. (2025), *RaceVLA*, [arXiv:2503.02572](https://arxiv.org/abs/2503.02572); Sautenkov et al. (2025), *UAV-VLA*, [arXiv:2501.05014](https://arxiv.org/abs/2501.05014).
4. Liu et al. (2023), *LIBERO*, [arXiv:2306.03310](https://arxiv.org/abs/2306.03310); Li et al. (2024), *Evaluating Real-World Robot Manipulation Policies in Simulation* (SimplerEnv), [arXiv:2405.05941](https://arxiv.org/abs/2405.05941).
5. Chi et al. (2024), *Universal Manipulation Interface*, [arXiv:2402.10329](https://arxiv.org/abs/2402.10329); Ma et al. (2023), *Eureka: Human-Level Reward Design via Coding Large Language Models*, [arXiv:2310.12931](https://arxiv.org/abs/2310.12931).
