# Chapter 7 Review

## Key ideas

1. A robot task is a **POMDP**: hidden state, observations, actions, dynamics, (often missing) reward. A VLA is a goal-conditioned, mostly Markov, chunked stochastic policy $\pi_\theta(\mathbf{A}_t\mid\mathbf{o}_t,\ell)$ in it.
2. **Behavioural cloning** is maximum likelihood on the *expert's* state distribution. It needs no reward and no trials, and it is how VLAs are trained.
3. **Covariate shift**: deployed on its own states, a BC policy's errors compound; the worst case grows as $T^2\epsilon$. In Lab 10, ten times more demonstrations of the same kind do not help (path error 0.060 → 0.061).
4. Fixes widen or correct the training distribution: **noise-injected demonstrations** (DART), **DAgger** (expert labels on learner states; linear in $T$), recovery data, **chunking** (fewer decisions), and stable **low-level control** under the policy.
5. **Policy gradients** reweight the policy's own actions by their advantage — weighted behavioural cloning. Baselines and critics reduce variance; **PPO** keeps updates near the data-collecting policy; **offline RL** must stay near the data.
6. **RL fine-tuning** of pretrained policies maximises reward minus a **KL anchor** to the BC policy. Reward-filtered and advantage-weighted BC work with any head. Sample complexity grows with the number of parameters trained: in Lab 10, a 4-parameter residual improves and a 4,000-parameter one gets worse at the same budget.

## Exercises

1. **Bellman and LQR.** For $\mathbf{s}_{t+1}=A\mathbf{s}_t+B\mathbf{a}_t$, cost $\mathbf{s}^\top Q\mathbf{s}+\mathbf{a}^\top R\mathbf{a}$ and $V(\mathbf{s})=\mathbf{s}^\top P\mathbf{s}$, derive the Riccati recursion from the Bellman optimality equation.
2. **Compounding error.** In the patrol task, BC adds a radial drift of about $V^2/2R$ per step when it flies exactly tangent. Estimate the path error after 400 steps and compare with Lab 10's measurement.
3. **Chunking.** A policy errs with probability $\epsilon=0.01$ per decision on expert states. Compare the compounding-error bound over $T=500$ control steps for chunk lengths $H_e=1,10,50$, assuming the per-chunk error is $\epsilon H_e^{1/2}$. When does chunking stop helping?
4. **Baselines.** Prove that subtracting any state-dependent baseline leaves the policy gradient unbiased. Why does a baseline that depends on the *action* break this?
5. **KL closed form.** Derive $\pi^\star\propto\pi_\text{ref}\exp(A/\beta)$ for the single-step KL-regularised objective.
6. **Privileged teacher.** Design a teacher–student pipeline for the depot arm: what the teacher observes in simulation, how it is trained, what the student observes, and where DAgger enters.
7. **Research question.** In a team of three drones, an operator corrects one drone's trajectory. How should the correction be added to a *shared* policy's dataset so that the other drones benefit, and what goes wrong if the drones' observations differ?

<!-- quiz: ch07 -->

## Reading assignments

1. [Ross, Gordon & Bagnell (2011), DAgger](../../papers.md#ross2011) — Sections 2–3 and Theorem 3.1.
2. [Schulman et al. (2017), PPO](../../papers.md#schulman2017) — Sections 3–5.
3. [Levine et al. (2020), Offline RL tutorial](../../papers.md#levine2020) — Sections 1–2 and 4.
4. Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html) — Chapter 13 (policy gradient methods).
