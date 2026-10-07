# 8. Vision-Language-Action Models

Everything so far converges here. A **VLA** is a pretrained vision-language backbone ([Chapter 5](../05-vlms/index.md)) with an action head that models action distributions ([Chapter 6](../06-generative/index.md)), trained mostly by behavioural cloning ([Chapter 7](../07-decision-making/index.md)), and run in closed loop on a robot. This chapter defines the paradigm formally, then takes its design decisions one at a time:

- how observations become tokens;
- what an "action" is;
- the two families of action heads, tokens and action experts;
- how chunks of actions are executed;
- what the closed loop around the model demands.

Lab 11 builds a toy VLA with every part and trains three action heads on the same depot task.

By the end you should be able to draw any VLA as backbone plus head, state the shape of every tensor between camera and actuator, predict its cost per action, and choose an action representation, head and execution scheme for a given robot.

## Learning objectives

1. Define a VLA precisely and classify architectures into token-based, action-expert, readout-head and dual-system families. ([8.1](definition.md))
2. Design the observation tokens — cameras, proprioception, language, history — and anticipate their failure modes. ([8.2](inputs.md))
3. Choose an action representation (joint vs end-effector, absolute vs delta, normalisation, rotations, embodiment) and the rate it runs at. ([8.3](action-spaces.md))
4. Build a token-based action head, explain FAST and parallel decoding, and compute its latency. ([8.4](action-tokens.md))
5. Attach a flow-matching action expert by shared attention, cross-attention or readout tokens, and explain how training couples it to the backbone. ([8.5](action-experts.md))
6. Execute chunks open loop, with receding horizon or with temporal ensembling, and reason about reactivity and mode consistency. ([8.6](chunking.md))
7. Build a latency budget for the whole control loop and recognise closed-loop failure modes. ([8.7](closed-loop.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [8.1](definition.md) | The VLA paradigm, formally | Lab 11 |
| [8.2](inputs.md) | Inputs: images, proprioception, language, history | Lab 11 |
| [8.3](action-spaces.md) | Action spaces and representations | Lab 11 |
| [8.4](action-tokens.md) | Autoregressive action tokens: binning, FAST, parallel decoding | Lab 11 |
| [8.5](action-experts.md) | Diffusion and flow-matching action experts | Lab 11 |
| [8.6](chunking.md) | Action chunking and temporal ensembling | Lab 11 |
| [8.7](closed-loop.md) | Closed-loop execution and control rates | Lab 11 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 11 — A toy VLA in the depot** (`labs/11_toy_vla/`). The ViT from Lab 07 sees a top-down depot with the robot in it, and the instruction says where to go. The same backbone is trained with an MSE head, an RT-2-style token head and a π0-style flow-matching expert. All three are evaluated in closed loop on ambiguous and unambiguous instructions, with chunking, receding horizon and temporal ensembling, and timed stage by stage.
