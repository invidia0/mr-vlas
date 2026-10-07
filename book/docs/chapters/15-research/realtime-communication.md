# 15.5 Communication-Aware and Real-Time Policies

VLAs are slow relative to the systems they control, and teams communicate over channels that are slow and lossy. Most research treats latency and communication as engineering details around a model. This section argues they should be part of the problem the model solves, and lists the open questions.

## Real-time policies

- **What exists.** Action chunking with temporal ensembling or asynchronous execution ([8.6](../08-vla/chunking.md)); asynchronous inference stacks (SmolVLA, [10.5](../10-case-studies/groot-smolvla.md)); real-time chunking for flow and diffusion policies, which conditions the next chunk on the actions that will execute during inference (Black et al., 2025); dual-system architectures that run a small fast policy under a large slow one ([10.5](../10-case-studies/groot-smolvla.md)); few-step flow sampling ([6.6](../06-generative/sampling-cost.md)).
- **Why hard.** Latency is a distribution, not a number ([11.1](../11-deployment/latency.md)); training data rarely contains the latency of deployment; and the consequences of delay depend on the task and the embodiment ([13.4](../13-uavs/direct-action.md)).
- **Research questions.**
    1. Can a VLA be trained with randomised observation delays (latency augmentation), so that it predicts actions for the time they will execute, and does this outperform post-hoc time alignment ([11.3](../11-deployment/serving.md))?
    2. What is the optimal split between a slow semantic model and a fast reactive one for a given hardware budget, and can the slow model's update rate adapt to how quickly the scene changes?
    3. How few flow steps, and how small an expert, can a VLA use before closed-loop success drops — and is the answer the same under latency?

## Communication-aware team policies

- **What exists.** Learned communication in multi-agent RL (CommNet and successors), event-triggered communication in control theory, consensus algorithms robust to loss ([14.4](../14-multi-robot/task-allocation.md)), and language messages between foundation-model agents ([14.3](../14-multi-robot/language-coordination.md)).
- **Research questions.**
    1. When should a robot running a VLA communicate? Can it learn to send messages only when they change a teammate's decision (event-triggered), under a bandwidth budget?
    2. How does coordination quality degrade with loss and latency for language messages versus structured claims, and can redundancy (re-broadcasting, as in Lab 16) be adapted to the measured channel?
    3. Can robots share compressed perception (visual tokens or embeddings rather than images) to build a team-level scene understanding at a fraction of the bandwidth ([14.2](../14-multi-robot/communication.md)), and what is lost?

## A first experiment with the course tools

In Lab 15, train the toy VLA with *latency augmentation*: during training, pair each action chunk with the observation from $L$ steps earlier, with $L$ drawn from the deployment latency distribution. Compare it with the plain policy plus time alignment (Lab 15, exercise 1) across latencies. In Lab 16, make the auction's re-broadcast period adapt to the observed loss rate and plot conflicts against messages.
