# Chapter 14 Review

## Key ideas

1. A team is a **Dec-POMDP**: shared reward, local observations, local policies. Optimal planning is NEXP-complete; practical systems use structure. Centralised designs coordinate best but scale and fail worst; **hierarchical / CTDE** designs — allocate or train centrally, act locally — are the practical middle; decentralised designs need communication to coordinate.
2. Single-robot VLAs placed together are **not a team**: with ambiguous team instructions, independent robots duplicate work (with uniform random choices, the chance $n$ robots split $n$ targets is $n!/n^n$; Lab 16's VLA does worse).
3. **Messages** trade size, expressiveness, interpretability and robustness. Natural language is compact, expressive and human-readable; channels are shared, lossy and range-limited. Messages enter VLAs as instruction text, extra tokens, or — most practically — through a task layer.
4. **Language-mediated coordination** (planner: SMART-LLM; negotiator: RoCo; communicator: CoELA; structure comparisons: Chen et al.) gains flexibility and readability, and needs grounding, constrained outputs, verification, and classical solvers for the optimisation.
5. **Task allocation**: ST-SR-IA is solved exactly (Hungarian, auction within $N\varepsilon$); greedy is suboptimal; CBBA decentralises with consensus; capabilities enter as infinite costs; VLMs supply tasks and costs, allocators solve, VLAs execute.
6. **Coverage control** (move to the weighted Voronoi centroid) shows how local rules give global guarantees; VLMs can supply the importance map. Emergent behaviour must be measured under perturbation.
7. **Team data** must record synchronised robots, messages, assignments and team outcomes; interactive fleet learning (Fleet-DAgger) allocates scarce human attention.

## Exercises

1. **Dec-POMDP.** Write the depot team task of Lab 16 as a Dec-POMDP: state, actions, observations, reward. Which information would a centralised policy have that the toy VLAs lack?
2. **Message budget.** Ten robots share a 2 Mbit/s channel at 70% efficiency. Compute the maximum rate per robot for (a) 64-byte state messages, (b) 300-byte language messages, (c) 30 kB compressed images. Which coordination schemes fit at 10 Hz?
3. **Verifier.** Extend the verifier of [14.3](language-coordination.md) to tasks requiring two robots (MR tasks) and to a precedence constraint ("inspect before picking").
4. **Auction by hand.** For costs $c=\begin{pmatrix}1&2\\1&4\end{pmatrix}$ and $\varepsilon=0.1$, run the auction algorithm step by step. Compare with greedy.
5. **Coverage with a VLM.** Design an importance map $\phi$ from a VLM's relevance scores for "find the person in the orange jacket" over a grid of image crops. How would you handle the VLM's false positives?
6. **Research question.** Propose a method to train a per-robot VLA that avoids duplicated targets *without* an allocator, using teammates' positions as extra tokens. What data would you need ([14.6](data.md)), and how would you evaluate it against Lab 16's allocators?

<!-- quiz: ch14 -->

## Reading assignments

1. Bernstein et al. (2002), *The Complexity of Decentralized Control of Markov Decision Processes*, [arXiv:1301.3836](https://arxiv.org/abs/1301.3836) (UAI 2000 version).
2. Gerkey & Matarić (2004), *A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems*, [IJRR 23(9)](https://doi.org/10.1177/0278364904045564).
3. Choi, Brunet & How (2009), *Consensus-Based Decentralized Auctions for Robust Task Allocation*, [IEEE T-RO](https://doi.org/10.1109/TRO.2009.2022423).
4. Cortés, Martínez, Karataş & Bullo (2004), *Coverage Control for Mobile Sensing Networks*, [arXiv:math/0212212](https://arxiv.org/abs/math/0212212).
5. Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO), [arXiv:2103.01955](https://arxiv.org/abs/2103.01955).
6. Kannan, Venkatesh & Min (2023), *SMART-LLM*, [arXiv:2309.10062](https://arxiv.org/abs/2309.10062); Mandi, Jain & Song (2023), *RoCo*, [arXiv:2307.04738](https://arxiv.org/abs/2307.04738); Chen et al. (2023), *Scalable Multi-Robot Collaboration with LLMs: Centralized or Decentralized Systems?*, [arXiv:2309.15943](https://arxiv.org/abs/2309.15943).
7. Zhang et al. (2023), *Building Cooperative Embodied Agents Modularly with Large Language Models* (CoELA), [arXiv:2307.02485](https://arxiv.org/abs/2307.02485).
8. Hoque et al. (2022), *Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision*, [arXiv:2206.14349](https://arxiv.org/abs/2206.14349).
