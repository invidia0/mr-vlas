# Reading List

This is a deliberately short list. Every paper here is either foundational, a clear milestone, or architecturally essential for understanding how VLAs came to be. Incremental papers are excluded on purpose. Each entry says why it matters, what problem it solved, what changed relative to prior work, which idea to focus on, and which section of the book it connects to.

**How to read them.** For each paper: read the abstract and look at the main figure; then read the method section with this book's corresponding section open; ignore most of the experiments on first reading except the headline comparison; finally, write down the shape of every tensor in the main architecture figure. If you can do that, you have understood the paper.

Entries are grouped by the part of the book they belong to.

---

## Part I — Foundations

### Rumelhart, Hinton & Williams (1986) — Learning representations by back-propagating errors {#rumelhart1986}

*Nature* 323, 533–536. [nature.com/articles/323533a0](https://www.nature.com/articles/323533a0)

- **Why it matters.** Popularised backpropagation for multilayer networks and showed that hidden layers learn useful internal representations.
- **Problem solved.** How to assign credit to hidden units — weights not directly connected to the output.
- **What changed.** Reverse-mode differentiation through a network made multilayer training practical.
- **Pay attention to.** The idea that hidden units discover features nobody designed. Everything in this book is a descendant.
- **Connects to.** [0.3 Backpropagation](chapters/00-foundations/training.md#backpropagation).

### Bengio, Ducharme, Vincent & Jauvin (2003) — A Neural Probabilistic Language Model {#bengio2003}

*JMLR* 3. [jmlr.org (PDF)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

- **Why it matters.** The first neural language model; introduced learned word embeddings trained jointly with the predictor.
- **Problem solved.** The curse of dimensionality of $n$-gram tables: unseen word combinations got zero probability.
- **What changed.** Words became dense vectors; similar words share statistical strength.
- **Pay attention to.** Section 2 (the architecture): embedding table → concatenation → MLP → softmax. It is a transformer without attention.
- **Connects to.** [0.6 Embeddings](chapters/00-foundations/embeddings.md), [0.7 n-gram models](chapters/00-foundations/sequence-models.md#n-gram-models-a-fixed-window).

### Mikolov, Chen, Corrado & Dean (2013) — Efficient Estimation of Word Representations in Vector Space (word2vec) {#mikolov2013}

[arXiv:1301.3781](https://arxiv.org/abs/1301.3781)

- **Why it matters.** Showed that embeddings trained with a trivially simple objective encode semantic relations as linear directions.
- **Problem solved.** Training word vectors on billions of words cheaply.
- **What changed.** Dropped the hidden layer of Bengio et al.; skip-gram and CBOW objectives.
- **Pay attention to.** The vector-arithmetic analogies, and *why* the distributional objective produces them.
- **Connects to.** [0.6 The distributional hypothesis](chapters/00-foundations/embeddings.md#why-geometry-becomes-meaningful-the-distributional-hypothesis); Lab 02.

### Kingma & Ba (2015) — Adam: A Method for Stochastic Optimization {#kingma2015}

[arXiv:1412.6980](https://arxiv.org/abs/1412.6980) — and its weight-decay fix, Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization*, [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).

- **Why it matters.** Adam/AdamW is the optimiser of essentially every transformer and VLA.
- **Problem solved.** Per-parameter step sizes for noisy, badly scaled gradients without manual tuning per layer.
- **What changed.** Combined momentum with RMS-normalised steps and bias correction.
- **Pay attention to.** Algorithm 1 and the bias-correction argument; in AdamW, why L2 regularisation and weight decay differ under adaptive scaling.
- **Connects to.** [0.3 Momentum and Adam](chapters/00-foundations/training.md#momentum-and-adam).

### Sennrich, Haddow & Birch (2016) — Neural Machine Translation of Rare Words with Subword Units (BPE) {#sennrich2016}

[arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

- **Why it matters.** Introduced byte-pair encoding to NLP; nearly all modern tokenizers descend from it.
- **Problem solved.** Out-of-vocabulary words in fixed word vocabularies.
- **What changed.** Learned subword units from corpus statistics instead of fixed words.
- **Pay attention to.** The ~10-line algorithm in the paper; compare with your implementation in Lab 01.
- **Connects to.** [0.5 Byte-pair encoding](chapters/00-foundations/tokens.md#how-does-it-work-byte-pair-encoding).

### Bahdanau, Cho & Bengio (2015) — Neural Machine Translation by Jointly Learning to Align and Translate {#bahdanau2015}

[arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

- **Why it matters.** Introduced attention.
- **Problem solved.** The fixed-size bottleneck of encoder–decoder RNNs.
- **What changed.** The decoder computes, at every step, a softmax-weighted average over all encoder states.
- **Pay attention to.** The alignment visualisations — learned attention weights as soft alignments between source and target words.
- **Connects to.** [0.7 Attention: let the decoder look back](chapters/00-foundations/sequence-models.md#attention-let-the-decoder-look-back).

### Vaswani et al. (2017) — Attention Is All You Need {#vaswani2017}

[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **Why it matters.** The transformer. Every model in this book is built from it.
- **Problem solved.** Sequential computation of RNNs prevented parallel training; long-range dependencies were hard to learn.
- **What changed.** Removed recurrence entirely; self-attention + position-wise FFN + residuals + normalisation; multi-head attention; sinusoidal positions.
- **Pay attention to.** Section 3 (every equation is derived in [Chapter 2](chapters/02-transformers/index.md)); Table 1 (complexity per layer vs RNNs); the footnote on why dot products are scaled by $1/\sqrt{d_k}$.
- **Connects to.** [2.1](chapters/02-transformers/attention.md), [2.2](chapters/02-transformers/multi-head.md), [2.4](chapters/02-transformers/block.md); Labs 03–04.

### Radford et al. (2019) — Language Models are Unsupervised Multitask Learners (GPT-2) {#radford2019}

[OpenAI (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- **Why it matters.** Showed that a decoder-only transformer trained only on next-token prediction performs many tasks zero-shot; introduced byte-level BPE.
- **Problem solved.** The need for a task-specific architecture and dataset per task.
- **What changed.** Pre-norm blocks, scale (1.5B parameters), byte-level BPE, tasks expressed as text.
- **Pay attention to.** Section 2 — "the task is part of the input". VLAs apply the same idea: the instruction specifies the task.
- **Connects to.** [1.1 Language models](chapters/01-multimodal-intro/language.md), [2.5](chapters/02-transformers/training.md).

### Brown et al. (2020) — Language Models are Few-Shot Learners (GPT-3) {#brown2020}

[arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

- **Why it matters.** Demonstrated in-context learning at scale (175B parameters): tasks specified by examples in the prompt, without weight updates.
- **Problem solved.** Fine-tuning for each new task.
- **What changed.** Mostly scale — which was the point.
- **Pay attention to.** Figure 1.2 and the zero/one/few-shot framing; Section 2.1 on model sizes, the reference table for parameter counts.
- **Connects to.** [2.7 Scaling](chapters/02-transformers/scaling.md).

### Kaplan et al. (2020) — Scaling Laws for Neural Language Models; Hoffmann et al. (2022) — Training Compute-Optimal LLMs (Chinchilla) {#scaling-laws}

[arXiv:2001.08361](https://arxiv.org/abs/2001.08361), [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

- **Why they matter.** Loss is a smooth power law in parameters, data and compute; Chinchilla corrected the optimal ratio to roughly 20 tokens per parameter.
- **Problem solved.** How to allocate a fixed compute budget between model size and data.
- **What changed.** Turned model design into a forecasting problem.
- **Pay attention to.** The $C\approx6ND$ compute approximation (derived in [2.7](chapters/02-transformers/scaling.md#compute)) and the Chinchilla isoFLOP curves. Ask yourself what the analogous laws for robot data are — this is an open question ([Chapter 15](chapters/15-research/index.md)).
- **Connects to.** [2.7 Scaling](chapters/02-transformers/scaling.md).

### Su et al. (2021) — RoFormer: Enhanced Transformer with Rotary Position Embedding {#su2021}

[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

- **Why it matters.** RoPE is the positional scheme of Llama, Qwen, Gemma — and therefore of the LLM backbones inside most VLAs.
- **Problem solved.** Encoding *relative* position inside the dot product without extra parameters.
- **What changed.** Positions as rotations of query/key pairs, so $\mathbf{q}_m\mathbf{k}_n^\top$ depends only on $m-n$.
- **Pay attention to.** Section 3.2 (the 2-D derivation, reproduced in [2.3](chapters/02-transformers/positions.md#rotary-position-embedding-rope)).
- **Connects to.** [2.3](chapters/02-transformers/positions.md); Lab 02.

### Elhage et al. (2021) — A Mathematical Framework for Transformer Circuits {#elhage2021}

[transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)

- **Why it matters.** The clearest account of *what a transformer computes*: the residual stream as a shared communication channel, attention heads as independent read/write operations, QK and OV circuits.
- **Problem solved.** Transformers as black boxes.
- **What changed.** A vocabulary for mechanistic interpretability.
- **Pay attention to.** "The residual stream" and "attention heads are independent and additive". This is the right mental model when you later inspect a VLA's internals.
- **Connects to.** [2.4 The residual stream](chapters/02-transformers/block.md#the-residual-stream); Lab 05.

### Dao et al. (2022) — FlashAttention {#dao2022}

[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

- **Why it matters.** Made long-sequence attention fast by being *IO-aware*; used inside essentially every modern training and inference stack.
- **Problem solved.** Attention was bound by GPU memory bandwidth, not FLOPs: the $T\times T$ score matrix was written to and read from slow memory.
- **What changed.** Tiled computation with an online softmax; the score matrix is never materialised. Exact, not approximate.
- **Pay attention to.** The online-softmax recurrence (Section 3.1) and the memory-hierarchy argument — the same reasoning governs VLA latency on embedded GPUs.
- **Connects to.** [2.6 Inference](chapters/02-transformers/inference.md#memory-bandwidth-and-flashattention).

---

## Part II — Seeing and reading

### He, Zhang, Ren & Sun (2016) — Deep Residual Learning for Image Recognition (ResNet) {#he2016}

[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

- **Why it matters.** Residual connections made very deep networks trainable; ResNets were the default vision backbone for years and are still the encoders of ACT and Diffusion Policy.
- **Problem solved.** Deeper plain networks trained *worse* (higher training error), an optimisation failure, not overfitting.
- **What changed.** Layers learn a correction $F(\mathbf{x})$ added to an identity path, $\mathbf{x}+F(\mathbf{x})$.
- **Pay attention to.** Section 3.1's argument that identity mappings are easier to learn as residuals; it is the origin of the transformer's residual stream.
- **Connects to.** [3.1](chapters/03-vision/cnn.md#the-canonical-architecture), [2.4](chapters/02-transformers/block.md#the-residual-stream).

### Dosovitskiy et al. (2021) — An Image is Worth 16x16 Words (ViT) {#dosovitskiy2021}

[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

- **Why it matters.** Showed a plain transformer on image patches matches or beats CNNs when pretrained at scale; turned images into tokens, which made VLMs and VLAs possible.
- **Problem solved.** Whether convolution's inductive biases are necessary for vision.
- **What changed.** Patches + linear embedding + learned positions + standard encoder; nothing image-specific beyond the patching.
- **Pay attention to.** Equations 1–4; Figure 3 (performance vs pretraining data size against ResNets); Figure 7 (learned position-embedding similarity showing emergent 2-D structure).
- **Connects to.** [3.2](chapters/03-vision/vit.md); Lab 06.

### Caron et al. (2021) — Emerging Properties in Self-Supervised Vision Transformers (DINO) {#caron2021}

[arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

- **Why it matters.** Self-distillation without labels yields ViT features whose attention segments objects; the basis of DINOv2.
- **Problem solved.** Learning strong visual features without labels and without negative pairs.
- **What changed.** Student predicts an EMA teacher's sharpened, centered output across multi-crop views.
- **Pay attention to.** Algorithm 1 (a page of pseudo-code) and the collapse ablation (centering vs sharpening).
- **Connects to.** [3.3](chapters/03-vision/self-supervised.md#dino-self-distillation-with-no-labels).

### Oquab et al. (2024) — DINOv2: Learning Robust Visual Features without Supervision {#oquab2024}

[arXiv:2304.07193](https://arxiv.org/abs/2304.07193) (TMLR 2024)

- **Why it matters.** The general-purpose frozen visual backbone; its dense features encode geometry, and OpenVLA fuses them with SigLIP.
- **Problem solved.** Making self-supervised features work off-the-shelf across tasks, at scale.
- **What changed.** Curated data (LVD-142M), an added patch-level masked objective, stabilisers, distillation into small models.
- **Pay attention to.** Section 3 (data curation pipeline) — data, more than the loss, was the lever.
- **Connects to.** [3.3](chapters/03-vision/self-supervised.md#dinov2-the-same-idea-engineered-at-scale); Lab 06.

### He et al. (2022) — Masked Autoencoders Are Scalable Vision Learners (MAE) {#he2022}

[arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

- **Why it matters.** Brought masked prediction to images efficiently; a common pretraining for robot-vision encoders.
- **Problem solved.** BERT-style masking had not worked as well for pixels as for words.
- **What changed.** 75% masking, encoder sees only visible patches, light decoder reconstructs pixels.
- **Pay attention to.** Figure 5 (masking-ratio ablation) and the efficiency argument of the asymmetric design.
- **Connects to.** [3.3](chapters/03-vision/self-supervised.md#mae-masked-autoencoders).

### Darcet et al. (2024) — Vision Transformers Need Registers {#darcet2024}

[arXiv:2309.16588](https://arxiv.org/abs/2309.16588)

- **Why it matters.** Explains the artifact tokens in large ViTs and removes them with a trivial change; cleaner dense features for downstream perception.
- **Problem solved.** High-norm, low-information tokens corrupting attention maps and patch features.
- **What changed.** A few extra learned tokens give the network scratch space.
- **Pay attention to.** Figures 1–2; then find the artifact in DINOv2-S yourself in Lab 06.
- **Connects to.** [3.2](chapters/03-vision/vit.md#the-cls-token-mean-pooling-and-registers).

### van den Oord, Li & Vinyals (2018) — Representation Learning with Contrastive Predictive Coding (InfoNCE) {#oord2018}

[arXiv:1807.03748](https://arxiv.org/abs/1807.03748)

- **Why it matters.** Introduced the InfoNCE loss and its mutual-information interpretation, which CLIP and most contrastive methods use.
- **Problem solved.** Learning representations by prediction without modelling every detail of the data.
- **What changed.** Classify the true future/paired sample among negatives instead of generating it.
- **Pay attention to.** Section 2.3: the loss and the bound $I\ge\log N-\mathcal{L}$.
- **Connects to.** [4.1](chapters/04-alignment/contrastive.md).

### Radford et al. (2021) — Learning Transferable Visual Models From Natural Language Supervision (CLIP) {#radford2021}

[arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

- **Why it matters.** Natural-language supervision at web scale; zero-shot recognition by text prompts; the template for vision encoders aligned with language.
- **Problem solved.** Fixed label sets and costly annotation.
- **What changed.** Two towers trained contrastively on 400 M image–text pairs; the classifier becomes text.
- **Pay attention to.** Figure 2 (contrastive vs predictive objectives' efficiency), Section 3.1.4 (prompt engineering and ensembling), and the robustness analysis.
- **Connects to.** [4.2](chapters/04-alignment/clip-siglip.md); Lab 07.

### Zhai et al. (2023) — Sigmoid Loss for Language Image Pre-Training (SigLIP) {#zhai2023}

[arXiv:2303.15343](https://arxiv.org/abs/2303.15343) — and its successor, Tschannen et al. (2025), *SigLIP 2*, [arXiv:2502.14786](https://arxiv.org/abs/2502.14786).

- **Why it matters.** The vision encoder of PaliGemma, π0, OpenVLA and SmolVLM.
- **Problem solved.** The batch-wide softmax of InfoNCE made large-batch training memory- and communication-heavy.
- **What changed.** An independent sigmoid per pair, with a learned bias; chunked implementation.
- **Pay attention to.** Section 3 and the role of the bias initialisation.
- **Connects to.** [4.2](chapters/04-alignment/clip-siglip.md#siglips-loss-step-by-step).

### Alayrac et al. (2022) — Flamingo: a Visual Language Model for Few-Shot Learning {#alayrac2022}

[arXiv:2204.14198](https://arxiv.org/abs/2204.14198)

- **Why it matters.** First VLM to show strong few-shot multimodal in-context learning from a frozen LLM.
- **Problem solved.** Connecting a frozen vision encoder and a frozen LLM without destroying either.
- **What changed.** Perceiver resampler + tanh-gated cross-attention layers initialised as the identity; interleaved image–text training data.
- **Pay attention to.** Figure 3 and Section 2.1; the gating initialisation.
- **Connects to.** [4.3](chapters/04-alignment/connectors.md#gated-cross-attention-leaving-the-lms-input-alone).

### Li et al. (2023) — BLIP-2 {#li2023}

[arXiv:2301.12597](https://arxiv.org/abs/2301.12597)

- **Why it matters.** The Q-Former: a small trained bridge that compresses an image into 32 tokens for a frozen LLM.
- **Problem solved.** Expensive end-to-end training of VLMs.
- **What changed.** A two-stage recipe: train the bridge on image–text objectives with the encoder frozen, then connect to a frozen LLM.
- **Pay attention to.** Figure 2 (the three objectives and their attention masks).
- **Connects to.** [4.3](chapters/04-alignment/connectors.md#learned-query-resamplers).

### Yuksekgonul et al. (2023) — When and Why Vision-Language Models Behave like Bags-of-Words {#yuksekgonul2023}

[arXiv:2210.01936](https://arxiv.org/abs/2210.01936)

- **Why it matters.** Diagnoses, from the training objective, why contrastive models ignore word order and binding — and fixes it with hard negatives.
- **Problem solved.** Benchmarks had not exposed compositional failures.
- **What changed.** The ARO benchmark; NegCLIP.
- **Pay attention to.** Sections 2–4: the argument that retrieval benchmarks can be solved without order information.
- **Connects to.** [4.5](chapters/04-alignment/failure-modes.md#bag-of-words-behaviour); Lab 07.

### Liu et al. (2023) — Visual Instruction Tuning (LLaVA) {#liu2023}

[arXiv:2304.08485](https://arxiv.org/abs/2304.08485) — and Liu et al. (2024), *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5), [arXiv:2310.03744](https://arxiv.org/abs/2310.03744).

- **Why it matters.** The simple, reproducible VLM recipe nearly all later VLMs and token-based VLAs follow.
- **Problem solved.** VLMs that follow free-form instructions about images, cheaply.
- **What changed.** Vision encoder + projector + LLM in one sequence; GPT-generated visual instruction data; two training stages.
- **Pay attention to.** LLaVA Section 4 (architecture and two-stage training); LLaVA-1.5 Table 1–2 (the MLP projector, higher resolution, data mixture).
- **Connects to.** [5.1](chapters/05-vlms/recipe.md), [5.2](chapters/05-vlms/training.md); Lab 08.

### Beyer et al. (2024) — PaliGemma: A versatile 3B VLM for transfer {#beyer2024}

[arXiv:2407.07726](https://arxiv.org/abs/2407.07726)

- **Why it matters.** The backbone of π0; a small, transferable VLM with a prefix-LM mask and location tokens.
- **Problem solved.** A base VLM designed to be fine-tuned for new tasks.
- **What changed.** SigLIP-So400m + Gemma-2B, prefix-LM attention, multitask pretraining with all parameters trained, resolution stages.
- **Pay attention to.** Section 3 (architecture, attention mask) and the ablations on freezing and the prefix-LM.
- **Connects to.** [5.3](chapters/05-vlms/small-vlms.md#paligemma-a-prefix-lm-with-location-tokens), [4.4](chapters/04-alignment/grounding.md#3-coordinates-as-tokens).

### Marafioti et al. (2025) — SmolVLM: Redefining Small and Efficient Multimodal Models {#marafioti2025}

[arXiv:2504.05299](https://arxiv.org/abs/2504.05299)

- **Why it matters.** Shows how far token budgets can be cut; the backbone family of SmolVLA.
- **Problem solved.** VLMs small enough for laptops and phones.
- **What changed.** Aggressive pixel shuffle, image splitting, small LMs with long context; careful study of the token budget.
- **Pay attention to.** The design-space findings on pixel shuffle and on vision/LM size balance.
- **Connects to.** [5.3](chapters/05-vlms/small-vlms.md#smolvlm-spending-as-few-tokens-as-possible); Lab 08.

### Ahn et al. (2022) — Do As I Can, Not As I Say (SayCan) {#ahn2022}

[arXiv:2204.01691](https://arxiv.org/abs/2204.01691)

- **Why it matters.** The first convincing combination of LLM knowledge with robot feasibility; the reference for LLM task planning.
- **Problem solved.** LLMs propose plans that the robot cannot execute in its current state.
- **What changed.** Score skills by LLM usefulness × learned affordance value.
- **Pay attention to.** Figure 2 and Section 3: the factorisation, and its dependence on a fixed skill library.
- **Connects to.** [5.5](chapters/05-vlms/to-action.md#rung-2-task-planner-over-a-skill-library).

### Liang et al. (2023) — Code as Policies {#liang2023}

[arXiv:2209.07753](https://arxiv.org/abs/2209.07753)

- **Why it matters.** Language models writing robot programs against perception and control APIs.
- **Problem solved.** Skill-list planners cannot express loops, conditions, arithmetic or new compositions.
- **What changed.** Few-shot prompted code generation, with hierarchical function definition.
- **Pay attention to.** The prompting structure and the examples of spatial-geometric reasoning in code.
- **Connects to.** [5.5](chapters/05-vlms/to-action.md#rung-3-code-as-policy).

---

## Part III — Acting

### Kingma & Welling (2014) — Auto-Encoding Variational Bayes (VAE) {#kingma2014}

[arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

- **Why it matters.** The ELBO plus the reparameterisation trick made latent-variable models trainable by backpropagation; the template for ACT's CVAE and the derivation of diffusion.
- **Problem solved.** Intractable posteriors in deep latent-variable models.
- **What changed.** An amortised encoder and a differentiable sampling path.
- **Pay attention to.** Sections 2.2–2.4: the estimator and the trick.
- **Connects to.** [6.3](chapters/06-generative/vae.md).

### Ho, Jain & Abbeel (2020) — Denoising Diffusion Probabilistic Models (DDPM) {#ho2020}

[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

- **Why it matters.** Made diffusion competitive; the basis of Diffusion Policy and diffusion action heads.
- **Problem solved.** Stable training of highly expressive generative models.
- **What changed.** The noise-prediction parameterisation and the simplified loss.
- **Pay attention to.** Section 2 and Algorithms 1–2 — they fit on half a page.
- **Connects to.** [6.4](chapters/06-generative/diffusion.md); Lab 09.

### Song, Meng & Ermon (2021) — Denoising Diffusion Implicit Models (DDIM) {#song2021}

[arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

- **Why it matters.** Deterministic, few-step sampling with the same trained network.
- **Problem solved.** Diffusion's hundreds of sampling steps.
- **What changed.** A non-Markovian forward process with the same marginals.
- **Pay attention to.** Section 4.1 (the sampler).
- **Connects to.** [6.4](chapters/06-generative/diffusion.md#faster-sampling-ddim), [6.6](chapters/06-generative/sampling-cost.md).

### Lipman et al. (2023) — Flow Matching for Generative Modeling {#lipman2023}

[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

- **Why it matters.** The training objective of π0, GR00T N1 and SmolVLA's action experts.
- **Problem solved.** Simulation-free training of continuous normalising flows.
- **What changed.** Regressing conditional velocities of chosen paths; the marginal field transports noise to data.
- **Pay attention to.** Sections 2–4, especially Theorem 2 (conditional and marginal objectives share gradients).
- **Connects to.** [6.5](chapters/06-generative/flow-matching.md); Labs 09, 11.

### Liu, Gong & Liu (2023) — Flow Straight and Fast (Rectified Flow) {#liu2023flow}

[arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

- **Why it matters.** Linear interpolation paths and *reflow* for straighter, few-step flows.
- **Problem solved.** Many-step sampling of ODE-based generative models.
- **What changed.** Straight paths and iterative straightening.
- **Pay attention to.** Section 2 and the reflow procedure.
- **Connects to.** [6.5](chapters/06-generative/flow-matching.md#sampling-step-counts-and-straightness).

### Zhao, Kumar, Levine & Finn (2023) — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT) {#zhao2023}

[arXiv:2304.13705](https://arxiv.org/abs/2304.13705)

- **Why it matters.** Action chunking with transformers and temporal ensembling; precise bimanual skills from ~50 demonstrations on cheap hardware (ALOHA).
- **Problem solved.** Compounding errors and multimodal human demonstrations in fine manipulation.
- **What changed.** Predict 100-step chunks with a CVAE-trained transformer; ensemble overlapping chunks.
- **Pay attention to.** Section IV and the ablations on chunk size and the CVAE.
- **Connects to.** [6.3](chapters/06-generative/vae.md#act-a-cvae-over-action-chunks), [8.6](chapters/08-vla/chunking.md).

### Chi et al. (2023) — Diffusion Policy {#chi2023}

[arXiv:2303.04137](https://arxiv.org/abs/2303.04137)

- **Why it matters.** Showed diffusion action heads handle multimodal demonstrations and outperform previous visuomotor policies; a standard baseline.
- **Problem solved.** Regression and mixture policies failing on multimodal, high-dimensional action sequences.
- **What changed.** Denoising action chunks conditioned on observations; receding-horizon execution.
- **Pay attention to.** Sections III–IV: chunk/execute horizons, visual encoder changes, the two denoiser architectures.
- **Connects to.** [6.4](chapters/06-generative/diffusion.md), [8.6](chapters/08-vla/chunking.md).

### Ross, Gordon & Bagnell (2011) — DAgger {#ross2011}

[arXiv:1011.0686](https://arxiv.org/abs/1011.0686)

- **Why it matters.** The reference analysis of compounding error and its fix by interactive data collection.
- **Problem solved.** Covariate shift in imitation learning.
- **What changed.** Reduction of imitation to no-regret online learning; expert labels on learner states.
- **Pay attention to.** Section 2 (the $T^2$ argument) and Theorem 3.1.
- **Connects to.** [7.3](chapters/07-decision-making/covariate-shift.md), [7.4](chapters/07-decision-making/dagger.md); Lab 10.

### Schulman et al. (2017) — Proximal Policy Optimization {#schulman2017}

[arXiv:1707.06347](https://arxiv.org/abs/1707.06347) — with GAE, Schulman et al. (2016), [arXiv:1506.02438](https://arxiv.org/abs/1506.02438).

- **Why it matters.** The default on-policy RL algorithm for simulated robot learning and for RLHF.
- **Problem solved.** Destructively large policy updates; poor sample reuse.
- **What changed.** A clipped probability-ratio objective allowing several epochs per batch.
- **Pay attention to.** Equation 7 and Section 5.
- **Connects to.** [7.5](chapters/07-decision-making/rl.md#ppo-small-safe-policy-updates).

### Levine, Kumar, Tucker & Fu (2020) — Offline Reinforcement Learning: Tutorial, Review, and Perspectives {#levine2020}

[arXiv:2005.01643](https://arxiv.org/abs/2005.01643)

- **Why it matters.** The clearest account of why learning from fixed datasets is hard — the regime of large robot datasets.
- **Problem solved.** (A survey) organising offline RL around distribution shift.
- **Pay attention to.** Sections 1–2 and 4.
- **Connects to.** [7.5](chapters/07-decision-making/rl.md#off-policy-and-offline-rl), [7.6](chapters/07-decision-making/rl-finetuning.md).

### Brohan et al. (2022) — RT-1: Robotics Transformer for Real-World Control at Scale {#brohan2022}

[arXiv:2212.06817](https://arxiv.org/abs/2212.06817)

- **Why it matters.** A transformer policy trained on 130 k real demonstrations across 700+ tasks; the direct predecessor of RT-2.
- **Problem solved.** Generalisation across many real-world tasks with one policy.
- **What changed.** FiLM-conditioned EfficientNet, TokenLearner, discretised action tokens.
- **Pay attention to.** Figure 3 (architecture) and the data-diversity results.
- **Connects to.** [1.3](chapters/01-multimodal-intro/combining.md), [3.4](chapters/03-vision/encoders-for-vlas.md), Chapter 10.

### Brohan et al. (2023) — RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control {#brohan2023}

[arXiv:2307.15818](https://arxiv.org/abs/2307.15818)

- **Why it matters.** Coined "VLA"; showed web-scale VLM knowledge transfers to robot control via action tokens.
- **Problem solved.** Policies that cannot generalise beyond robot-data semantics.
- **What changed.** Actions as text tokens in a VLM; co-fine-tuning with web data.
- **Pay attention to.** Section 3 (action tokenisation, co-fine-tuning) and the emergent-capability evaluations.
- **Connects to.** [8.1](chapters/08-vla/definition.md), [8.4](chapters/08-vla/action-tokens.md).

### Open X-Embodiment Collaboration (2024) — Open X-Embodiment: Robotic Learning Datasets and RT-X Models {#oxe2024}

[arXiv:2310.08864](https://arxiv.org/abs/2310.08864)

- **Why it matters.** Pooled 60+ datasets from 22 robots into one format; the training data of Octo and OpenVLA.
- **Problem solved.** Fragmented, incompatible robot datasets.
- **Pay attention to.** The dataset analysis and the positive-transfer results.
- **Connects to.** Chapter 9.

### Octo Model Team (2024) — Octo: An Open-Source Generalist Robot Policy {#octo2024}

[arXiv:2405.12213](https://arxiv.org/abs/2405.12213)

- **Why it matters.** An open, flexible generalist policy with a diffusion head and readout tokens, designed for fine-tuning to new robots.
- **Pay attention to.** The architecture figure (tokenizers, readouts, block-wise masks).
- **Connects to.** [8.5](chapters/08-vla/action-experts.md#3-readout-tokens-octo), Chapter 10.

### Kim et al. (2024) — OpenVLA: An Open-Source Vision-Language-Action Model {#kim2024}

[arXiv:2406.09246](https://arxiv.org/abs/2406.09246) — and its fine-tuning follow-up, Kim, Finn & Liang (2025), *OpenVLA-OFT*, [arXiv:2502.19645](https://arxiv.org/abs/2502.19645).

- **Why it matters.** The open 7 B token-based VLA, and a careful study of design decisions.
- **What changed.** SigLIP+DINOv2 fused encoder, Llama 2 backbone, 256-bin action tokens, OXE training; OFT adds parallel decoding, chunking and continuous regression.
- **Pay attention to.** Section 3.4 (design decisions: resolution, fine-tuning the encoder, epochs) and OFT's speed/quality ablations.
- **Connects to.** [8.4](chapters/08-vla/action-tokens.md), Chapters 9–11.

### Black et al. (2024) — π0: A Vision-Language-Action Flow Model for General Robot Control {#black2024}

[arXiv:2410.24164](https://arxiv.org/abs/2410.24164) — and π0.5 (Physical Intelligence, 2025), [arXiv:2504.16054](https://arxiv.org/abs/2504.16054).

- **Why it matters.** The flow-matching action-expert design for dexterous, high-rate control on a VLM backbone.
- **What changed.** PaliGemma backbone + separate action-expert weights with a block-wise mask; 50-step chunks by flow matching.
- **Pay attention to.** Section IV and Figure 3; π0.5's co-training for open-world generalisation.
- **Connects to.** [8.5](chapters/08-vla/action-experts.md#1-shared-attention-with-a-block-wise-mask-0), Chapter 10.

### Pertsch et al. (2025) — FAST: Efficient Action Tokenization for VLA Models {#pertsch2025}

[arXiv:2501.09747](https://arxiv.org/abs/2501.09747)

- **Why it matters.** Makes token-based VLAs viable for high-frequency, long-chunk control.
- **What changed.** DCT + quantisation + BPE compresses action chunks.
- **Pay attention to.** Section 4 and the comparison with binning.
- **Connects to.** [8.4](chapters/08-vla/action-tokens.md#fast-compress-the-chunk-before-tokenising).

### NVIDIA (2025) — GR00T N1: An Open Foundation Model for Generalist Humanoid Robots {#gr00t2025}

[arXiv:2503.14734](https://arxiv.org/abs/2503.14734)

- **Why it matters.** An open dual-system VLA: a VLM for understanding and a diffusion-transformer action head at a higher rate.
- **Pay attention to.** The System 1 / System 2 split and the data pyramid (real, synthetic, human video).
- **Connects to.** [8.5](chapters/08-vla/action-experts.md#dual-system-timing), Chapter 10.

### Shukor et al. (2025) — SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics {#shukor2025}

[arXiv:2506.01844](https://arxiv.org/abs/2506.01844)

- **Why it matters.** A ~0.5 B VLA trainable and runnable on consumer hardware, built on community data.
- **What changed.** SmolVLM2 backbone with skipped layers, 64 tokens per image, a flow-matching expert with interleaved cross/self-attention, asynchronous inference.
- **Pay attention to.** The efficiency design choices and asynchronous execution.
- **Connects to.** [5.3](chapters/05-vlms/small-vlms.md), [8.5](chapters/08-vla/action-experts.md), Chapter 10.

### Hu et al. (2022) — LoRA: Low-Rank Adaptation of Large Language Models {#hu2022}

[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

- **Why it matters.** The standard way to adapt a large pretrained model with few trainable parameters; OpenVLA and many VLA fine-tuning recipes use it.
- **What changed.** A frozen weight plus a trainable low-rank update $BA$, with $B$ initialised to zero, mergeable after training (no inference cost).
- **Pay attention to.** Which matrices get adapters, and the rank ablations.
- **Connects to.** [9.4](chapters/09-training/finetuning.md), Lab 13.

### Khazatsky et al. (2024) — DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset {#khazatsky2024}

[arXiv:2403.12945](https://arxiv.org/abs/2403.12945)

- **Why it matters.** Diverse real-world manipulation data collected with one standardised platform across many labs and scenes — diversity rather than repetition.
- **Pay attention to.** The collection protocol and the analysis of how scene diversity affects generalisation.
- **Connects to.** [9.2](chapters/09-training/demonstrations.md), [9.3](chapters/09-training/datasets.md), [9.6](chapters/09-training/data-scaling.md).

---

## Part IV — Deploying

### Dettmers et al. (2023) — QLoRA: Efficient Finetuning of Quantized LLMs {#dettmers2023}

[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

- **Why it matters.** Introduced the 4-bit NormalFloat (NF4) data type and showed that fine-tuning through a 4-bit frozen model with LoRA adapters matches 16-bit fine-tuning — the basis of most low-memory VLA adaptation and deployment.
- **Pay attention to.** NF4 (quantiles of a normal distribution), double quantisation of the scales.
- **Connects to.** [11.2](chapters/11-deployment/memory-quantization.md), [9.4](chapters/09-training/finetuning.md).

### Macenski et al. (2022) — Robot Operating System 2: Design, Architecture, and Uses in the Wild {#macenski2022}

[arXiv:2211.07752](https://arxiv.org/abs/2211.07752)

- **Why it matters.** The design rationale of the middleware nearly every research robot runs: DDS, QoS, executors, lifecycle nodes.
- **Pay attention to.** QoS policies and why real-time behaviour needs explicit design.
- **Connects to.** [11.4](chapters/11-deployment/ros2.md), [Appendix B](appendix/ros2-px4.md).

### Ames et al. (2019) — Control Barrier Functions: Theory and Applications {#ames2019}

[arXiv:1903.11199](https://arxiv.org/abs/1903.11199)

- **Why it matters.** The tutorial reference for CBFs: minimally invasive safety filters with forward-invariance guarantees, solvable as small QPs at control rates.
- **Pay attention to.** The condition $\dot h\ge-\alpha(h)$, the CBF-QP, and the relation to Lyapunov functions.
- **Connects to.** [12.4](chapters/12-embodied/interfaces.md), [13.5](chapters/13-uavs/safety.md), Labs 15–16.

### Kaufmann et al. (2023) — Champion-level drone racing using deep reinforcement learning {#kaufmann2023}

[Nature 620, 982–987](https://www.nature.com/articles/s41586-023-06419-4)

- **Why it matters.** A learned policy beat human world champions in real drone racing — and shows what direct learned control of a drone required: a small, fast policy, extensive simulation, residual models of real dynamics, and a tightly controlled track.
- **Pay attention to.** The perception/policy split and how the sim-to-real gap was closed.
- **Connects to.** [13.4](chapters/13-uavs/direct-action.md).

### Black, Galliker & Levine (2025) — Real-Time Execution of Action Chunking Flow Policies {#black2025rtc}

[arXiv:2506.07339](https://arxiv.org/abs/2506.07339)

- **Why it matters.** Makes asynchronous chunk execution smooth for flow and diffusion VLAs without retraining, by freezing the actions that will execute during inference and inpainting the rest.
- **Pay attention to.** How the inference delay enters the sampling procedure.
- **Connects to.** [8.6](chapters/08-vla/chunking.md), [11.3](chapters/11-deployment/serving.md), [15.5](chapters/15-research/realtime-communication.md).

---

## Part V — Research

### Bernstein, Zilberstein & Immerman (2000) — The Complexity of Decentralized Control of Markov Decision Processes {#bernstein2000}

[arXiv:1301.3836](https://arxiv.org/abs/1301.3836)

- **Why it matters.** Defined Dec-POMDPs and proved that optimal decentralised planning is NEXP-complete — why multi-robot systems rely on structure rather than exact solutions.
- **Connects to.** [14.1](chapters/14-multi-robot/dec-pomdp.md).

### Gerkey & Matarić (2004) — A Formal Analysis and Taxonomy of Task Allocation in Multi-Robot Systems {#gerkey2004}

[IJRR 23(9)](https://doi.org/10.1177/0278364904045564)

- **Why it matters.** The ST/MT, SR/MR, IA/TA taxonomy that organises multi-robot task allocation, and the link to classical assignment problems.
- **Connects to.** [14.4](chapters/14-multi-robot/task-allocation.md).

### Choi, Brunet & How (2009) — Consensus-Based Decentralized Auctions for Robust Task Allocation {#choi2009}

[IEEE T-RO](https://doi.org/10.1109/TRO.2009.2022423)

- **Why it matters.** CBBA: decentralised allocation by bundle construction and consensus, robust to inconsistent information and changing topology — the reference algorithm for communication-limited teams.
- **Connects to.** [14.4](chapters/14-multi-robot/task-allocation.md), Lab 16.

### Cortés, Martínez, Karataş & Bullo (2004) — Coverage Control for Mobile Sensing Networks {#cortes2004}

[arXiv:math/0212212](https://arxiv.org/abs/math/0212212)

- **Why it matters.** Distributed gradient laws (move to the weighted Voronoi centroid) with convergence guarantees — the template for scalable local coordination.
- **Connects to.** [14.5](chapters/14-multi-robot/coverage-scalability.md).

### Yu et al. (2022) — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO) {#yu2022}

[arXiv:2103.01955](https://arxiv.org/abs/2103.01955)

- **Why it matters.** Showed that PPO with centralised value functions is a strong baseline for cooperative MARL — the CTDE recipe in its simplest form.
- **Connects to.** [14.1](chapters/14-multi-robot/dec-pomdp.md), [7.5](chapters/07-decision-making/rl.md).

### Mandi, Jain & Song (2023) — RoCo: Dialectic Multi-Robot Collaboration with Large Language Models {#mandi2023}

[arXiv:2307.04738](https://arxiv.org/abs/2307.04738)

- **Why it matters.** LLM agents, one per robot arm, negotiate sub-task plans and waypoints in dialogue, validated by environment feedback before a motion planner executes them.
- **Pay attention to.** The feedback loop between dialogue and environment checks.
- **Connects to.** [14.3](chapters/14-multi-robot/language-coordination.md).

### Kannan, Venkatesh & Min (2023) — SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models {#kannan2023}

[arXiv:2309.10062](https://arxiv.org/abs/2309.10062)

- **Why it matters.** An LLM as central team planner: task decomposition, coalition formation and allocation through programmatic prompts.
- **Connects to.** [14.3](chapters/14-multi-robot/language-coordination.md), [14.4](chapters/14-multi-robot/task-allocation.md).

### Zhang et al. (2023) — Building Cooperative Embodied Agents Modularly with Large Language Models (CoELA) {#zhang2023coela}

[arXiv:2307.02485](https://arxiv.org/abs/2307.02485)

- **Why it matters.** Decentralised embodied agents that decide what to communicate and do, built from perception, memory, communication, planning and execution modules around an LLM.
- **Connects to.** [14.2](chapters/14-multi-robot/communication.md), [14.3](chapters/14-multi-robot/language-coordination.md).

### Hoque et al. (2022) — Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision {#hoque2022}

[arXiv:2206.14349](https://arxiv.org/abs/2206.14349)

- **Why it matters.** Formalised learning from a fleet of robots supervised by a few humans, and the allocation of human attention by robot uncertainty and risk.
- **Connects to.** [14.6](chapters/14-multi-robot/data.md).

### Ren et al. (2023) — Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners (KnowNo) {#ren2023}

[arXiv:2307.01928](https://arxiv.org/abs/2307.01928)

- **Why it matters.** Conformal prediction turns an LLM planner's uncertainty into an "ask for help" rule with a statistical guarantee on task success.
- **Connects to.** [11.5](chapters/11-deployment/monitoring.md), [15.4](chapters/15-research/safety-uncertainty.md).

### Lykov et al. (2025) — CognitiveDrone: A VLA Model and Evaluation Benchmark for Real-Time Cognitive Task Solving and Reasoning in UAVs {#lykov2025}

[arXiv:2503.01378](https://arxiv.org/abs/2503.01378)

- **Why it matters.** One of the first VLAs for drones with a benchmark; shows the gain from a separate VLM reasoning step before high-frequency control.
- **Connects to.** [13.3](chapters/13-uavs/architectures.md), [15.2](chapters/15-research/uavs.md).
