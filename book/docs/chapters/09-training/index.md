# 9. How VLAs Are Trained

Part IV turns from *what* a VLA is to *how it is made to work* on a robot — training, deployment, and the robot stack around it. This chapter is about training: the sequence of stages from web pretraining to your robot's demonstrations, how demonstrations are collected and stored, the datasets that generalists are built from, how to adapt a billion-parameter model on one GPU with LoRA, how to evaluate a policy so that the numbers mean something, and why data diversity and the absence of shortcuts matter more than data volume.

By the end you should be able to plan a post-training project for your own robot: what data to collect and how, in which format, what to fine-tune and with what memory, how to co-train against forgetting, and how many evaluation episodes you need to conclude anything.

## Learning objectives

1. Lay out the training pipeline of a VLA, the role of each stage, and how co-training counters catastrophic forgetting. ([9.1](stages.md))
2. Compare demonstration sources — teleoperation interfaces, scripted and privileged experts, simulation, human video — and record demonstrations that stay usable. ([9.2](demonstrations.md))
3. Describe OXE, BridgeData, DROID and LIBERO, and read and write the LeRobot dataset format. ([9.3](datasets.md))
4. Derive LoRA, compute the memory of full fine-tuning vs LoRA vs QLoRA, and follow the adaptation recipe for a new robot. ([9.4](finetuning.md))
5. Design closed-loop evaluations along explicit generalisation axes and report success rates with confidence intervals. ([9.5](evaluation.md))
6. Identify shortcuts in a dataset, distinguish diversity from volume, and state what is known about scaling for robot data. ([9.6](data-scaling.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [9.1](stages.md) | Training stages and co-training | Lab 13 |
| [9.2](demonstrations.md) | Demonstration collection and teleoperation | Lab 13 |
| [9.3](datasets.md) | Datasets and the LeRobot format | Lab 13 |
| [9.4](finetuning.md) | Fine-tuning and LoRA | Lab 13 |
| [9.5](evaluation.md) | Evaluating VLAs | Labs 11, 13 |
| [9.6](data-scaling.md) | Data quality, scaling and distribution shift | Lab 11 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 13 — Data and adaptation** (`labs/13_data_adaptation/`): depot demonstrations written and read in the LeRobot format; the toy VLA adapted to new object kinds by full fine-tuning and by LoRA at several ranks, with and without co-training, measuring success, memory and forgetting.
