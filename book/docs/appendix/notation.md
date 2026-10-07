# Appendix C — Notation and Shape Conventions

These conventions are fixed across the whole book and all labs. When an equation in a chapter uses a symbol from this page, it means exactly this.

## General mathematics

| Symbol | Meaning |
|---|---|
| $x$, $\alpha$ | scalars (lower-case italic) |
| $\mathbf{x}$ | vector; **row vector** $\mathbf{x}\in\mathbb{R}^{1\times d}$ when it is a token/feature, so that $\mathbf{x}W$ is a linear map (this matches PyTorch, where the last dimension is the feature dimension) |
| $X$, $W$ | matrices (upper-case) |
| $X_{i,:}$, $X_{:,j}$ | $i$-th row, $j$-th column |
| $X^\top$ | transpose |
| $\odot$ | element-wise (Hadamard) product |
| $\langle \mathbf{a}, \mathbf{b}\rangle = \mathbf{a}\mathbf{b}^\top$ | dot product of two row vectors |
| $\lVert\mathbf{x}\rVert_2$ | Euclidean norm |
| $[\mathbf{a};\mathbf{b}]$ | concatenation along the feature dimension |
| $\mathbb{E}_{x\sim p}[f(x)]$ | expectation of $f(x)$ when $x$ is drawn from $p$ |
| $\mathcal{N}(\boldsymbol\mu,\Sigma)$ | Gaussian with mean $\boldsymbol\mu$, covariance $\Sigma$ |
| $\operatorname{softmax}(\mathbf{z})_i = e^{z_i}/\sum_j e^{z_j}$ | softmax over the last dimension unless stated otherwise |
| $\nabla_\theta \mathcal{L}$ | gradient of loss $\mathcal{L}$ w.r.t. parameters $\theta$ |
| $\log$ | natural logarithm |

## Tensor dimension symbols

| Symbol | Meaning | Typical values |
|---|---|---|
| $B$ | batch size | 1 (robot inference) – 4096 (pretraining) |
| $T$ | sequence length (number of tokens, query side) | 16 – 8192 |
| $S$ | key/value sequence length (in cross-attention, the length of the *other* sequence) | |
| $d$ (also $d_\text{model}$) | model width: size of each token vector | 128 (toy) – 4096 (7B LLM) |
| $h$ | number of attention heads | 4 – 32 |
| $d_h = d/h$ | per-head width; $d_k$ (query/key width) and $d_v$ (value width) are both $d_h$ unless stated | 64 – 128 |
| $d_\text{ff}$ | hidden width of the feed-forward network | $\approx 4d$ (or $\tfrac{8}{3}d$ with SwiGLU) |
| $L$ | number of transformer layers | 2 (toy) – 32 (7B) |
| $V$ | vocabulary size | 256 (bytes) – 256 000 |
| $H, W, C$ | image height, width, channels | $224, 224, 3$ |
| $P$ | patch side length in pixels | 14 or 16 |
| $N_p = HW/P^2$ | number of image patches/tokens | 196 – 1024 |
| $d_a$ | action dimension | 7 (arm: 6-DoF end-effector delta + gripper), 4 (quadrotor: velocity + yaw rate) |
| $H_a$ | action-chunk horizon (number of future actions predicted at once) | 1 – 50 |
| $N$ | number of robots in a team | |

In code, shapes are annotated as comments using the same letters: `x  # (B, T, d)`.

## Learning and control

| Symbol | Meaning |
|---|---|
| $\theta$ | all learnable parameters of a model |
| $\mathcal{L}(\theta)$ | loss function |
| $\eta$ | learning rate |
| $\mathcal{D}$ | dataset |
| $t$ | discrete time step of the robot/control loop (not to be confused with $T$, a sequence length) |
| $\mathbf{s}_t$ | true (possibly unobservable) state |
| $\mathbf{o}_t$ | observation at time $t$ (images + proprioception) |
| $\mathbf{q}_t$ | proprioceptive state (joint angles, end-effector pose, vehicle attitude, …) |
| $\ell$ | language instruction (a string; after tokenization, a sequence of token ids) |
| $\mathbf{a}_t\in\mathbb{R}^{d_a}$ | action at time $t$ |
| $\mathbf{A}_t = (\mathbf{a}_t,\dots,\mathbf{a}_{t+H_a-1})$ | action chunk |
| $\pi_\theta(\mathbf{a}\mid\mathbf{o},\ell)$ | policy: a conditional distribution over actions |
| $r_t$, $\gamma$ | reward and discount factor (reinforcement learning) |
| $\tau$ | diffusion / flow time (in [0, 1] or $\{1..K\}$), kept distinct from robot time $t$ |
| $H_e$ | execution horizon: actions executed from each chunk before re-querying the policy |
| $\Delta t$, $\delta$ | control period; end-to-end latency (observation to applied action) |

## Deployment, safety and teams

| Symbol | Meaning |
|---|---|
| $h(\mathbf{x})$ | control barrier function; the safe set is $\{h\ge0\}$ ([12.4](../chapters/12-embodied/interfaces.md)) |
| $\alpha$ | CBF rate: enforce $\dot h\ge-\alpha h$ |
| $\mathbf{u}_\text{prop}$, $\mathbf{u}^\ast$ | proposed and filtered command |
| $T_\text{wd}$ | watchdog timeout |
| $\mathbf{g}_k$ | goal or waypoint proposed by a goal-level VLA ([13.3](../chapters/13-uavs/architectures.md)) |
| $i, j$ | robot index $i=1..N$; task index $j=1..M$ |
| $\mathbf{o}_i$, $\mathbf{a}_i$, $\pi_i$ | robot $i$'s local observation, action and policy (Dec-POMDP, [14.1](../chapters/14-multi-robot/dec-pomdp.md)) |
| $\mathbf{m}_{i\to j}$, $\mathcal{N}_i(t)$ | message from robot $i$ to $j$; $i$'s communication neighbourhood |
| $c_{ij}$, $x_{ij}$, $p_j$ | allocation cost, assignment variable, auction price ([14.4](../chapters/14-multi-robot/task-allocation.md)) |
| $V_i$, $\phi$ | Voronoi cell of robot $i$; importance density ([14.5](../chapters/14-multi-robot/coverage-scalability.md)) |
