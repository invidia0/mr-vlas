# 7. Sequential Decision Making: Imitation and Reinforcement Learning

Chapter 6 gave us models of action *distributions*. This chapter is about the loop those actions live in. A robot's actions change what it sees next, so training on one distribution of states and deploying on another is the normal case, not a corner case. Here we build the theory of learning policies: MDPs and POMDPs; behavioural cloning and why its errors compound; DAgger and interactive imitation; the policy-gradient, actor-critic and offline-RL tools; and how reinforcement learning fine-tunes large pretrained policies. Lab 10 measures every claim on a patrol task that runs hundreds of episodes in parallel on a CPU.

By the end you should be able to formulate a robot task as a (PO)MDP, explain why a VLA with low training loss can fail on the robot, choose between more demonstrations, interactive corrections and RL, and derive the policy gradient that RL fine-tuning uses.

## Learning objectives

1. Define MDPs, POMDPs, returns, value functions, advantages and the Bellman equation, and state the simplifications built into a VLA's policy signature. ([7.1](mdp.md))
2. Write behavioural cloning as maximum likelihood on the expert's state distribution, and list the data properties that matter. ([7.2](behavior-cloning.md))
3. Derive the $T^2\epsilon$ compounding-error bound and evaluate mitigations, from noise injection to action chunking. ([7.3](covariate-shift.md))
4. State DAgger, its linear-in-$T$ guarantee, and its practical variants with human and privileged experts. ([7.4](dagger.md))
5. Derive the policy gradient, explain baselines, GAE, PPO's clipped objective and the difficulty of offline RL. ([7.5](rl.md))
6. Explain KL-regularised fine-tuning of a pretrained policy, the approaches available for token and flow heads, and reward sources for robots. ([7.6](rl-finetuning.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [7.1](mdp.md) | MDPs, POMDPs and policies | Lab 10 |
| [7.2](behavior-cloning.md) | Behavioural cloning | Lab 10 |
| [7.3](covariate-shift.md) | Covariate shift and compounding error | Lab 10 |
| [7.4](dagger.md) | DAgger and interactive imitation | Lab 10 |
| [7.5](rl.md) | Policy gradients, actor-critic, offline RL | Lab 10 |
| [7.6](rl-finetuning.md) | RL fine-tuning of large pretrained policies | Lab 10 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 10 — Imitation and RL on a patrol task** (`labs/10_imitation_rl/`). A drone follows a circular route in gusty wind. BC drifts off the route, and ten times more demonstrations do not help. Noise-injected demonstrations and DAgger fix it. Residual RL on top of the frozen cloned policy fixes it from a reward alone, but only when few parameters are trained.
