# 15.1 VLAs for Multi-Robot Systems and Decentralised Robotics

Chapter 14 showed that single-robot VLAs placed together are not a team, and that today's working designs put a classical allocator or a verified language planner above per-robot VLAs. This section lays out what is open beyond that: VLAs that model their teammates, foundation models that run decentralised, coordination learned from team data, and evaluation that measures teams rather than robots.

## Open problems

### 1. Teammate-aware VLAs

A per-robot VLA conditions on its own camera, its state and an instruction. A teammate-aware VLA would also condition on what it knows about the others: positions, claimed tasks, recent messages, intentions. It would act so that the *team* succeeds — taking the crate nobody else is heading for, yielding in a corridor, covering a teammate's blind spot.

- **Why hard.** There is almost no team demonstration data ([14.6](../14-multi-robot/data.md)). Inputs scale with team size. The policy must generalise across team compositions it never saw. Evaluating coordination needs many team episodes.
- **What exists.** Multi-agent RL learns such behaviour in simulation for small, fixed teams (CTDE; [14.1](../14-multi-robot/dec-pomdp.md)). VLAs with extra state tokens ([8.2](../08-vla/inputs.md)) provide the architectural slot. Lab 16 provides a minimal testbed.
- **Research questions.**
    1. Can a VLA learn to avoid duplicated targets from team demonstrations with teammates' positions as tokens, and how many team episodes does it need compared with single-robot data plus an allocator?
    2. Which teammate representation generalises across team sizes: a set of tokens with attention, a pooled embedding, or a language summary ("robot 2 is going to the blue crate")?
    3. Does co-training on single-robot and team data preserve single-robot skill while adding coordination ([9.1](../09-training/stages.md))?

### 2. Decentralised foundation models

Running a large model per robot is expensive; running one centrally is fragile. Between these lie designs where robots share computation opportunistically, or run small local models and query larger shared ones when uncertain.

- **Research questions.**
    1. A robot runs a small local VLA and escalates to a shared large model when its uncertainty ([11.5](../11-deployment/monitoring.md)) is high. What escalation rule minimises team cost (compute, communication, failures) under a bandwidth budget?
    2. How should a team allocate a shared GPU server among robots with different deadlines ([11.3](../11-deployment/serving.md))? Is deadline-aware batching enough, or should robots adapt their chunk lengths to server load?
    3. Can robots share *visual tokens* instead of images to build a team-level scene representation within a radio budget ([14.2](../14-multi-robot/communication.md))?

### 3. Language as the coordination layer, made reliable

Language-mediated coordination ([14.3](../14-multi-robot/language-coordination.md)) is flexible but unreliable. The most promising direction combines language models that *state* the problem with verified solvers that *solve* it.

- **Research questions.**
    1. For a team instruction, how often does an LLM-extracted task/cost/constraint formulation, solved exactly, beat an LLM that writes the assignment directly? Where do the extraction errors come from?
    2. Can the verifier's feedback (rejected assignments) be used to fine-tune the language planner, as safety-filter interventions could for VLAs ([12.4](../12-embodied/interfaces.md))?
    3. What is the right message format between robots running VLAs — free text, a constrained grammar, or structured claims with a language field — under loss and latency?

### 4. Evaluating teams

Single-robot VLA evaluation is already statistically demanding ([9.5](../09-training/evaluation.md)). Team evaluation multiplies the variance: success depends on every robot, the communication channel, and their interactions.

- **Research questions.**
    1. Which team-level metrics (success, duplicates, makespan, messages, near-collisions, robustness to one failed robot) predict real deployment value, and how many episodes are needed to compare two coordination methods at a given effect size?
    2. How should a benchmark perturb teams (message loss, a faulty teammate, a late joiner) to expose brittle emergent behaviour ([14.5](../14-multi-robot/coverage-scalability.md))?

## A first experiment with the course tools

Extend Lab 16. Collect team demonstrations in the 2-D depot in which an allocator decides targets. Add a "teammates" token to `TinyVLA` (the positions of the other robots, padded to a maximum team size), and train with and without it. Measure duplicated targets with independent execution at team sizes 2–4, with Wilson intervals, against the centralised auction. That is research question 1.1 in miniature.
