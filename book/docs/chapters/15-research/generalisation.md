# 15.3 Long-Horizon Autonomy, Generalisation and Data Efficiency

The headline limitation of today's VLAs is not a missing component. They work for minutes on tasks resembling their data, and degrade over longer horizons, in new places, and with few demonstrations. This section groups three tightly linked problems: acting coherently over long horizons, generalising beyond the training distribution (including from simulation to reality), and learning from little robot data.

## Long-horizon autonomy and embodied reasoning

Most VLAs are reactive: $\pi(\mathbf{A}_t\mid\mathbf{o}_t,\ell)$ with no memory beyond the current observation ([7.1](../07-decision-making/mdp.md)). Long tasks need memory (what has been done, where things were seen), sub-goal structure, and recovery from failures.

- **What exists.** Hierarchical outputs inside one model — predict the subtask in language, then its actions (π0.5, [10.4](../10-case-studies/pi0.md)). External task planners over VLA skills (SayCan-style, [12.2](../12-embodied/hierarchy.md)). Dual systems with slow reasoning and fast control ([10.5](../10-case-studies/groot-smolvla.md)).
- **Why hard.** Long-horizon demonstrations are expensive and rare; errors compound across sub-tasks ([7.3](../07-decision-making/covariate-shift.md)); evaluation needs long episodes.
- **Research questions.**
    1. What memory representation lets a VLA act over tens of minutes — a history of key frames, a language log of completed subtasks, or a spatial map with semantic labels — and what does each cost in tokens per step ([8.2](../08-vla/inputs.md))?
    2. Can a VLA detect that its current sub-task has failed and choose a recovery, using its own uncertainty ([11.5](../11-deployment/monitoring.md)) as a trigger?
    3. How should reasoning ("think, then act") be budgeted in time so that the control loop never starves ([8.7](../08-vla/closed-loop.md))?

## Generalisation and sim-to-real

Lab 11 showed how easily a VLA learns a shortcut instead of the intended skill (it ignored the instruction), and Lab 12 measured a sim-to-sim gap between two renderings of the same task. Real deployment adds lighting, clutter, new objects, new robots and new users.

- **What exists.** Diverse, multi-embodiment pretraining data (OXE, DROID; [9.3](../09-training/datasets.md)), co-training with web data ([9.1](../09-training/stages.md)), domain randomisation and photorealistic simulation ([Appendix A](../../appendix/mujoco.md)), benchmark suites with real-to-sim correlation (SimplerEnv) and controlled distribution shifts (LIBERO).
- **Research questions.**
    1. Which axes of diversity (scenes, objects, instructions, embodiments) buy the most generalisation per demonstration, for a fixed budget ([9.6](../09-training/data-scaling.md))?
    2. Can shortcut learning be detected *before* deployment — for example, by probing whether actions change when the instruction changes (the test Lab 11's failure analysis performs)?
    3. For mobile and aerial robots, does a goal-level output transfer from simulation better than an action-level output, because the dynamics gap is absorbed by classical layers ([13.3](../13-uavs/architectures.md))?

## Data efficiency

A new task on a new robot usually starts with tens to hundreds of demonstrations ([9.4](../09-training/finetuning.md)). Reducing that number is a direct multiplier on what robots can be taught.

- **What exists.** Parameter-efficient fine-tuning (LoRA, Lab 13), strong pretrained backbones, action chunking and generative heads that model demonstrators' variability, handheld data-collection devices that decouple data from robots (e.g. UMI, Chi et al., 2024), and LLM-written rewards for RL fine-tuning (e.g. Eureka, Ma et al., 2023).
- **Research questions.**
    1. In Lab 13, the smallest LoRA ranks adapted best, and LoRA forgot the original task as much as full fine-tuning. Do both findings hold for real VLAs on genuinely new skills (not only new objects), and does the best rank predict how "new" a task is?
    2. How much can RL fine-tuning in simulation ([7.6](../07-decision-making/rl-finetuning.md)) replace real demonstrations for the last mile of reliability?
    3. Can a team of robots share adaptation — one robot's corrections improving all — without destabilising the shared policy ([14.6](../14-multi-robot/data.md))?

## A first experiment with the course tools

Repeat Lab 13's adaptation with 30, 100, 300 and 1000 cone episodes, for full fine-tuning and LoRA at rank 4, and plot success on the new and the original task against data size. Then add 20% data from the MuJoCo depot (Lab 12) and test whether the mixture improves transfer to MuJoCo without hurting the 2-D world.
