# Chapter 0 Review

## Key ideas

1. Deep learning operates on **tensors**; tracking shapes is the main debugging tool, and memory (bytes per parameter × parameters) is the first constraint on robots.
2. A **neural network** is a composition of affine maps and non-linearities; its **inductive bias** determines how much data it needs.
3. **Training** is minibatch stochastic optimisation (AdamW) of an empirical risk, with gradients from **backpropagation** — reverse-mode differentiation, i.e. the discrete adjoint method.
4. Losses are **negative log-likelihoods**: MSE ↔ unimodal Gaussian (regression to the mean); cross-entropy ↔ categorical (any distribution over $V$ outcomes).
5. **Tokenization** turns text into integers; tokens are the unit of cost. Subword BPE is the standard.
6. **Embeddings** map symbols (and, later, patches, states and actions) into a shared vector space whose geometry carries meaning.
7. Sequence models evolved from fixed windows to **RNNs** (sequential, bottlenecked) to **attention** and the **transformer** (parallel, all-pairs).
8. A VLA = vision encoder + language model backbone + action head; every component is a chapter of this book.

## Exercises

1. **Shapes.** Write the shape of every intermediate tensor when a batch of 16 depot observations — two $224\times224$ RGB cameras, an 8-D proprioceptive vector, and a 20-token instruction — is embedded into width $d=512$, using $P=14$ patches. How many tokens per example?
2. **Backprop by hand.** Derive $\partial\mathcal{L}/\partial\mathbf{z}$ for $\mathcal{L}=-\log\operatorname{softmax}(\mathbf{z})_y$ and verify it equals $\operatorname{softmax}(\mathbf{z})-\mathbf{e}_y$.
3. **Adam as a preconditioner.** For the quadratic $\tfrac12\theta^\top\operatorname{diag}(\lambda_1,\dots,\lambda_n)\theta$ with exact gradients, show that Adam with $\beta_1=\beta_2=0$ and $\epsilon=0$ takes the step $\eta\operatorname{sign}(\theta)$, independent of the curvatures $\lambda_i$. Compare with gradient descent, whose step along coordinate $i$ is $\eta\lambda_i\theta_i$. What does this say about Adam on ill-conditioned problems, and why does it still need a decaying learning rate to converge?
4. **Mixture NLL.** Write the negative log-likelihood of a two-component 1-D Gaussian mixture $p(y)=\pi\mathcal{N}(y;\mu_1,\sigma^2)+(1-\pi)\mathcal{N}(y;\mu_2,\sigma^2)$. Show that, unlike MSE, its minimiser on the bimodal grasp data of [0.4](probability.md#sampling) puts $\mu_1,\mu_2$ at the two handles.
5. **BPE.** Continue the worked example of [0.5](tokens.md#worked-example) for three more merges.
6. **Tokens as cost.** A VLA runs at 5 Hz on two cameras with 256 tokens each, plus 30 text and state tokens. Ignoring everything except token count, by what factor does its per-step cost change if you add a third camera? If you double image resolution at fixed patch size?
7. **Research question.** Seven robots each send their observation tokens to a central VLA. Which of the costs in this chapter (memory, tokens, sequence length) grows with the number of robots, and how? Keep your answer; [Chapter 14](../14-multi-robot/index.md) returns to it.

<!-- quiz: ch00 -->

## Reading assignments

1. [Bengio et al. (2003)](../../papers.md#bengio2003) — the neural language model; read Section 2 and compare with the transformer of [0.8](first-look.md#transformers).
2. [Sennrich et al. (2016)](../../papers.md#sennrich2016) — BPE; check your Lab 01 implementation against the paper's algorithm.
3. [Bahdanau et al. (2015)](../../papers.md#bahdanau2015) — attention as alignment; look at the alignment figures.
4. Optional: [Kingma & Ba (2015)](../../papers.md#kingma2015) — Adam; Algorithm 1 only.
