# Chapter 12 Review

## Key ideas

1. A VLA sees cameras and a compact **estimated** state at its own slow rate; fast proprioception feeds estimators and controllers. Every input costs tokens and data, every omitted one must be handled elsewhere; frames, calibration and time synchronisation make state meaningful.
2. Robots are a **hierarchy** of loops — task, skill, motion, control, safety — with rates increasing downwards by roughly 5–10× per layer. VLAs usually occupy the skill layer; VLMs the task layer; hierarchical VLAs span both; for fast or dangerous vehicles the VLA should produce goals or waypoints.
3. **Learn** components that are hard to model by hand, slow enough for the policy, recoverable when they fail and backed by data; **keep classical** the fast, well-modelled and safety-critical ones (estimation, low-level control, limits).
4. The **interface** decides what the lower layer can check. Post-process every output (un-normalise, convert, time-align, smooth), then filter it: saturation, geofences, rate limits, watchdogs, and **control barrier functions**, which minimally change a command so that $\dot h\ge-\alpha h$ and the safe set is never left.

## Exercises

1. **Routing.** For a quadrotor with a downward camera, a forward camera, IMU, GNSS, barometer and a companion computer running a VLA at 2 Hz, list which estimates the VLA receives and which sensors it never sees. Justify each choice by rate.
2. **Rate separation.** In the two-rate simulation of [12.2](hierarchy.md), reduce the PD gains until the controller no longer reaches the target within one policy period. Plot the trajectories and explain what the policy now experiences.
3. **Learn or keep.** Apply the procedure of [12.3](learn-or-classical.md) to: (a) door opening with a mobile manipulator; (b) landing a drone on a moving platform; (c) sorting parcels by label text. Which components would you learn?
4. **CBF by hand.** For $h(\mathbf{x})=\lVert\mathbf{x}-\mathbf{c}\rVert^2-r^2$ with $\mathbf{c}=(0,0)$, $r=1$, $\mathbf{x}=(1.5,0)$, $\alpha=2$ and $\mathbf{u}_\text{prop}=(-2,0.5)$, compute the filtered velocity. Which component was removed, and why does the robot not stop?
5. **Deadlock.** Show that if $\mathbf{u}_\text{prop}$ is parallel to $\mathbf{c}-\mathbf{x}$, the filtered velocity goes to zero as the robot reaches the boundary. Propose two remedies and their costs.
6. **Research question.** Learned policies sometimes "fight" a safety filter. Design an experiment that measures how often a VLA's proposals are modified by the filter, and propose how this signal could be used for training (e.g. as a penalty in RL fine-tuning, [7.6](../07-decision-making/rl-finetuning.md)) without making the policy depend on the filter.

<!-- quiz: ch12 -->

## Reading assignments

1. [Ahn et al. (2022), SayCan](../../papers.md#ahn2022) and [Liang et al. (2023), Code as Policies](../../papers.md#liang2023) — foundation models at the task layer over classical skills.
2. Ames et al. (2019), *Control Barrier Functions: Theory and Applications*, [arXiv:1903.11199](https://arxiv.org/abs/1903.11199) — Sections I–III.
3. Siciliano & Khatib (eds.), *Springer Handbook of Robotics* (2nd ed., 2016) — chapters on control architectures and force control, for the classical layers beneath a VLA.
4. Thrun, Burgard & Fox, *Probabilistic Robotics* (MIT Press, 2005) — Chapter 3 (Gaussian filters).
