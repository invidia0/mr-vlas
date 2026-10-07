# 15.6 From Research Question to Paper

The previous sections list questions; this one is about turning one into a result others can trust. VLA research has specific traps: expensive and noisy evaluation, baselines that are hard to reproduce, simulation numbers that do not predict real behaviour, and claims that rest on a handful of trials. The checklist below is distilled from the methods used throughout this book.

## A checklist

### 1. State the question as a comparison

"Does X improve Y under condition Z, compared with W?" — for example, "does a teammate token reduce duplicated targets (Y) for teams of 2–4 robots (Z) compared with independent VLAs and with a centralised auction (W)?". Name the effect size that would matter before running anything.

### 2. Choose baselines that could win

Include the strongest *simple* baseline, not only prior learned methods: a classical allocator for coordination ([14.4](../14-multi-robot/task-allocation.md)), a scripted controller for navigation, the MSE head for action generation ([6.1](../06-generative/multimodality.md)), full fine-tuning for adaptation ([9.4](../09-training/finetuning.md)). Lab 16's auction and Lab 11's regression head are examples of baselines that are hard to beat.

### 3. Control the data

Match training data, compute and tuning effort across methods. Report how demonstrations were collected, filtered and split; separate scenes, objects and instructions between training and test, so that shortcut learning cannot pass for generalisation (Lab 11).

### 4. Evaluate with enough trials, paired

Report success with confidence intervals; evaluate all methods on the *same* episodes (seeds, initial conditions) and use paired tests ([9.5](../09-training/evaluation.md), Lab 13). Decide the number of episodes from the effect size you need to detect. Report failure modes, not only rates (Lab 11's failure analysis).

### 5. Stress the claim

Test under the conditions that would break it: latency (Lab 15), message loss (Lab 16), distribution shift (Labs 12–13), new team sizes. Include ablations that remove the proposed component.

### 6. Separate simulation and reality

Say what was simulated. If claiming real-world relevance, show either real-robot results or evidence that the simulation predicts reality for this metric (e.g. real-to-sim correlation). For aerial robots, show the safety envelope used ([13.5](../13-uavs/safety.md)).

### 7. Make it reproducible

Release code, configurations, seeds, model checkpoints with their normalisation statistics, and data in a standard format (LeRobot, [9.3](../09-training/datasets.md)). Log versions of every dependency ([11.5](../11-deployment/monitoring.md)).

## A template for a first project

| Step | With the course tools |
|---|---|
| question | one of the research questions of [15.1](multi-robot.md)–[15.5](realtime-communication.md) |
| environment | 2-D depot (Labs 11, 15, 16) for speed; MuJoCo depot (Lab 12) for physics and rendering; PX4 SITL ([Appendix B](../../appendix/ros2-px4.md)) for UAVs |
| model | the toy VLA (`vlacourse.vla.TinyVLA`) for controlled studies; SmolVLA (Lab 14) for realism |
| data | scripted experts with `collect`, stored in the LeRobot format (Lab 13) |
| evaluation | `evaluate` with Wilson intervals and paired seeds; failure analysis |
| reporting | the checklist above |

## Why it matters for VLAs and robots

The field moves fast, and many claims do not survive careful evaluation. Results that state a comparison, beat strong simple baselines, report uncertainty and are reproducible are the ones that last — and, for robots that operate near people, the only ones that should change practice.
