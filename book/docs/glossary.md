# Glossary

Short definitions with a link to the section where each term is defined and developed. 

| Term | Definition | Defined in |
|---|---|---|
| **[CLS] token** | Learned token prepended to a ViT's patches whose output summarises the image. | [3.2](chapters/03-vision/vit.md#the-cls-token-mean-pooling-and-registers) |
| **Action chunk** | A sequence of $H_a$ future actions predicted at once, $\mathbf{A}_t=(\mathbf{a}_t,\dots,\mathbf{a}_{t+H_a-1})$. | [Notation](appendix/notation.md#learning-and-control), Ch 8 |
| **Action expert** | A separate set of transformer weights that generates continuous action chunks (flow matching or diffusion) conditioned on the VLM backbone's features. | [8.5](chapters/08-vla/action-experts.md) |
| **Action tokens** | Discretised action values written by a language-model backbone as tokens (RT-2, OpenVLA; FAST compresses chunks first). | [8.4](chapters/08-vla/action-tokens.md) |
| **Activation function** | Element-wise non-linearity between layers (ReLU, GELU, SiLU, tanh). | [0.2](chapters/00-foundations/neural-networks.md#why-the-non-linearity-is-essential) |
| **adaLN** | Adaptive normalisation: a FiLM-style scale/shift of a normalised feature, predicted from a condition (e.g. diffusion time). | [1.3](chapters/01-multimodal-intro/combining.md#how-to-fuse-four-mechanisms) |
| **Adam / AdamW** | Optimiser with per-parameter steps from running gradient moments; AdamW adds decoupled weight decay. | [0.3](chapters/00-foundations/training.md#momentum-and-adam) |
| **Affordance** | The probability that a skill can succeed from the current state (SayCan's value function). | [5.5](chapters/05-vlms/to-action.md#rung-2-task-planner-over-a-skill-library) |
| **Arithmetic intensity** | FLOPs per byte moved; decides whether an operation is compute- or bandwidth-bound. | [2.6](chapters/02-transformers/inference.md#arithmetic-intensity) |
| **Asynchronous execution** | Computing the next action chunk while the current one executes, so that inference latency causes staleness instead of pauses. | [6.6](chapters/06-generative/sampling-cost.md), [11.3](chapters/11-deployment/serving.md) |
| **Attention** | Softmax-weighted average of values, with weights from query–key similarity. | [2.1](chapters/02-transformers/attention.md#what-is-it) |
| **Attention mask** | Additive $-\infty$/0 (or boolean) matrix restricting which keys each query may read: padding, causal, prefix. | [2.1](chapters/02-transformers/attention.md#masking) |
| **Auction algorithm** | Assignment by bidding: robots bid for tasks at current prices, raising them by their margin plus ε; within Nε of optimal and naturally distributed. | [14.4](chapters/14-multi-robot/task-allocation.md) |
| **Autoregressive model** | Factorises $p(x_{1:T})=\prod_tp(x_t\mid x_{<t})$ and generates one element at a time. | [0.7](chapters/00-foundations/sequence-models.md#the-autoregressive-factorisation) |
| **Backpropagation** | Reverse-mode automatic differentiation of a scalar loss through the computational graph. | [0.3](chapters/00-foundations/training.md#backpropagation) |
| **Batch** | Independent examples processed together along the leading tensor dimension $B$. | [0.1](chapters/00-foundations/tensors.md#how-does-it-work-shapes-as-a-type-system) |
| **Behavioural cloning (BC)** | Supervised learning of a policy from demonstrated observation–action pairs. | [7.2](chapters/07-decision-making/behavior-cloning.md) |
| **BPE (byte-pair encoding)** | Subword tokenizer built by repeatedly merging the most frequent adjacent symbol pair. | [0.5](chapters/00-foundations/tokens.md#how-does-it-work-byte-pair-encoding) |
| **Broadcasting** | Virtual repetition of size-1 dimensions so element-wise ops apply to differently shaped tensors. | [0.1](chapters/00-foundations/tensors.md#element-wise-operations-and-broadcasting) |
| **Causal mask** | Mask letting position $i$ attend only to positions $\le i$. | [2.1](chapters/02-transformers/attention.md#masking) |
| **CBBA** | Consensus-Based Bundle Algorithm: decentralised task allocation by bundle construction and consensus on winning bids. | [14.4](chapters/14-multi-robot/task-allocation.md) |
| **CLIP** | Two-tower image–text model trained with InfoNCE on 400 M web pairs; zero-shot recognition by prompts. | [4.2](chapters/04-alignment/clip-siglip.md) |
| **Co-training** | Mixing earlier data (web, other robots, the original task) into later training stages to keep general capabilities. | [9.1](chapters/09-training/stages.md) |
| **Conformal prediction** | Distribution-free calibration that turns model scores into prediction sets with guaranteed coverage; used to decide when a robot asks for help. | [15.4](chapters/15-research/safety-uncertainty.md) |
| **Connector (projector, resampler)** | Module mapping vision-encoder features to image tokens in an LLM's embedding space. | [4.3](chapters/04-alignment/connectors.md) |
| **Context window** | Maximum number of tokens a model conditions on. | [1.1](chapters/01-multimodal-intro/language.md#context-window) |
| **Contrastive learning / InfoNCE** | Training embeddings so matched pairs score higher than in-batch negatives; InfoNCE is the softmax form. | [4.1](chapters/04-alignment/contrastive.md) |
| **Control barrier function (CBF)** | A function $h$ with safe set $h\ge0$; enforcing $\dot h\ge-\alpha h$ keeps the system safe. A CBF filter minimally modifies commands to satisfy it. | [12.4](chapters/12-embodied/interfaces.md) |
| **Convolution** | Sliding a small shared kernel over a feature map; local, weight-shared, translation-equivariant. | [3.1](chapters/03-vision/cnn.md#what-is-it) |
| **Cosine similarity** | Dot product of $\ell_2$-normalised vectors. | [0.6](chapters/00-foundations/embeddings.md#measuring-similarity) |
| **Covariate shift** | The learner visits states the expert's demonstrations never covered, so small errors compound over time. | [7.3](chapters/07-decision-making/covariate-shift.md) |
| **Coverage control** | Distributed control in which each robot moves towards the weighted centroid of its Voronoi cell. | [14.5](chapters/14-multi-robot/coverage-scalability.md) |
| **Cross-attention** | Attention with queries from one sequence and keys/values from another. | [2.2](chapters/02-transformers/multi-head.md#cross-attention) |
| **Cross-entropy** | $H(p,q)=-\sum p\log q$; the NLL loss for categorical predictions. | [0.4](chapters/00-foundations/probability.md#entropy-cross-entropy-and-kl-divergence) |
| **CTDE** | Centralised training, decentralised execution: train with global information, act on local information. | [14.1](chapters/14-multi-robot/dec-pomdp.md) |
| **DAgger** | Dataset aggregation: roll out the learner, have the expert label the visited states, retrain on all data. | [7.4](chapters/07-decision-making/dagger.md) |
| **Dec-POMDP** | Decentralised POMDP: several agents with local observations and policies maximise a shared reward. | [14.1](chapters/14-multi-robot/dec-pomdp.md) |
| **Decoding strategy** | Rule mapping logits to the next token: greedy, beam, temperature, top-$k$, top-$p$. | [1.1](chapters/01-multimodal-intro/language.md#decoding) |
| **Defence in depth** | Independent protection layers, each able to stop a hazard, arranged so that all must fail for harm to occur. | [13.5](chapters/13-uavs/safety.md) |
| **Diffusion model** | A generative model that learns to reverse a gradual noising process; generates by iterative denoising. | [6.4](chapters/06-generative/diffusion.md) |
| **DINO / DINOv2** | Self-distillation of a ViT with an EMA teacher, centering and sharpening; DINOv2 scales it with curated data. | [3.3](chapters/03-vision/self-supervised.md) |
| **Domain randomisation** | Varying simulator parameters (lighting, colours, masses) during training so that a policy becomes robust to them. | [Appendix A](appendix/mujoco.md#6-randomisation-and-reproducibility) |
| **Embedding** | Learned map from discrete symbols (or other inputs) to vectors in $\mathbb{R}^d$. | [0.6](chapters/00-foundations/embeddings.md#what-is-it) |
| **Empirical risk** | Average loss over the training set. | [0.3](chapters/00-foundations/training.md#what-is-it) |
| **ENU / NED** | Local frames: east-north-up (ROS) and north-east-down (PX4); confusing them mirrors and inverts motion. | [13.2](chapters/13-uavs/px4-offboard.md) |
| **Equivariance / invariance** | $f(T x)=T f(x)$ vs $f(T x)=f(x)$ for a transformation $T$ such as a shift. | [3.1](chapters/03-vision/cnn.md#why-does-it-exist-three-assumptions-about-images) |
| **Exposure bias** | Mismatch between training on true prefixes and generating from the model's own outputs. | [0.7](chapters/00-foundations/sequence-models.md#the-autoregressive-factorisation) |
| **FFN (feed-forward network)** | Position-wise MLP inside each transformer block. | [2.4](chapters/02-transformers/block.md#the-feed-forward-network) |
| **FiLM** | Feature-wise linear modulation: $\boldsymbol\gamma(\mathbf{c})\odot\mathbf{h}+\boldsymbol\beta(\mathbf{c})$. | [1.3](chapters/01-multimodal-intro/combining.md#how-to-fuse-four-mechanisms) |
| **FlashAttention** | Exact, tiled, IO-aware attention that never materialises the score matrix. | [2.6](chapters/02-transformers/inference.md#flashattention) |
| **Fleet learning** | Pooling many robots' experience and human interventions into shared datasets and policies. | [14.6](chapters/14-multi-robot/data.md) |
| **Flight controller** | The vehicle's real-time computer (e.g. PX4 on a Pixhawk): estimator, cascaded controllers and failsafes. | [13.1](chapters/13-uavs/stack.md) |
| **FLOPs** | Floating-point operations; a matmul of $m\times k$ by $k\times n$ is $2mkn$. | [0.1](chapters/00-foundations/tensors.md#matrix-multiplication) |
| **Flow matching** | Training a velocity field that transports noise to data along simple (e.g. straight) paths; generation integrates it. | [6.5](chapters/06-generative/flow-matching.md) |
| **Fusion (early/late)** | Where in a network modalities are combined. | [1.3](chapters/01-multimodal-intro/combining.md#where-to-fuse-early-late-intermediate) |
| **Geofence** | A boundary the vehicle must not cross, enforced by the companion's filter and independently by the flight controller. | [13.5](chapters/13-uavs/safety.md) |
| **GQA / MQA** | Grouped-/multi-query attention: several query heads share key/value heads. | [2.2](chapters/02-transformers/multi-head.md#multi-query-and-grouped-query-attention) |
| **Gradient descent / SGD** | Iterative parameter updates along the negative (minibatch) gradient. | [0.3](chapters/00-foundations/training.md#gradient-descent) |
| **Grounding problem** | How symbols acquire meaning tied to the physical world. | [1.3](chapters/01-multimodal-intro/combining.md#the-grounding-problem) |
| **Hallucination** | A VLM asserting content that is not in the image, driven by its language prior. | [5.4](chapters/05-vlms/spatial.md) |
| **Hard negative** | A negative example similar to the positive; dominates the contrastive gradient. | [4.1](chapters/04-alignment/contrastive.md#the-gradient-pull-the-positive-push-the-negatives) |
| **Image tokens** | Projected visual features occupying positions in a language model's input sequence. | [5.1](chapters/05-vlms/recipe.md) |
| **Inductive bias** | Assumptions that make a learner prefer some functions over others. | [0.2](chapters/00-foundations/neural-networks.md#universal-approximation-and-what-it-does-not-say) |
| **KL divergence** | $D_\text{KL}(p\Vert q)=\sum p\log(p/q)\ge0$. | [0.4](chapters/00-foundations/probability.md#entropy-cross-entropy-and-kl-divergence) |
| **KV cache** | Stored keys/values of past positions, enabling incremental decoding. | [2.6](chapters/02-transformers/inference.md#the-kv-cache) |
| **Language model (LM / LLM)** | Distribution over token sequences; an LLM is a large decoder-only transformer LM. | [1.1](chapters/01-multimodal-intro/language.md#what-is-it), [0.8](chapters/00-foundations/first-look.md#large-language-models) |
| **Latency budget** | Time allowances for every stage of the control loop, at a high percentile, that must fit the deadline. | [11.1](chapters/11-deployment/latency.md) |
| **LayerNorm / RMSNorm** | Per-token normalisation over features (RMSNorm omits mean-centring). | [2.4](chapters/02-transformers/block.md#normalisation) |
| **LeRobot dataset** | Robot data format: Parquet tables (state, action, timestamps), MP4 videos per camera, and metadata with fps, features and statistics. | [9.3](chapters/09-training/datasets.md) |
| **Linear probe** | A linear classifier trained on frozen features to measure what they encode. | [3.3](chapters/03-vision/self-supervised.md#what-is-it) |
| **Location tokens** | Vocabulary entries for discretised coordinates, letting a language model output boxes as text. | [4.4](chapters/04-alignment/grounding.md#3-coordinates-as-tokens) |
| **Logit lens** | Applying the output head to intermediate residual-stream states. | [2.4](chapters/02-transformers/block.md#the-residual-stream) |
| **Logits** | Unnormalised scores whose softmax gives class probabilities. | [0.4](chapters/00-foundations/probability.md#maximum-likelihood) |
| **LoRA** | Low-rank adaptation: a frozen weight plus a trainable update $BA$ of rank $r$, with $B$ initialised to zero. | [9.4](chapters/09-training/finetuning.md) |
| **Loss mask** | Per-token weights selecting which predictions contribute to the loss. | [2.5](chapters/02-transformers/training.md#loss-masking-what-the-model-is-trained-to-say) |
| **MAE** | Masked autoencoder: reconstruct 75% hidden patches; encoder sees only visible ones. | [3.3](chapters/03-vision/self-supervised.md#mae-masked-autoencoders) |
| **Maximum likelihood (MLE)** | Choosing parameters that maximise the probability of observed data. | [0.4](chapters/00-foundations/probability.md#maximum-likelihood) |
| **MLP** | Multilayer perceptron: alternating affine maps and activations. | [0.2](chapters/00-foundations/neural-networks.md#what-is-it) |
| **Modality** | A type of signal (text, image, proprioception, action). | [1.3](chapters/01-multimodal-intro/combining.md#what-is-it) |
| **Modality gap** | Image and text embeddings of a contrastive model occupying separate cones of the sphere. | [4.5](chapters/04-alignment/failure-modes.md#the-modality-gap) |
| **MRTA** | Multi-robot task allocation; classified as ST/MT, SR/MR, IA/TA (Gerkey & Matarić). | [14.4](chapters/14-multi-robot/task-allocation.md) |
| **Multi-head attention** | $h$ parallel attention heads in $d/h$-dim subspaces, combined by $W_O$. | [2.2](chapters/02-transformers/multi-head.md#what-is-it) |
| **Offboard mode** | PX4 mode that tracks setpoints streamed by a companion computer while a heartbeat continues. | [13.2](chapters/13-uavs/px4-offboard.md) |
| **Open-vocabulary detection** | Detecting boxes for any category named in text at run time. | [4.4](chapters/04-alignment/grounding.md#2-open-vocabulary-detectors) |
| **Patch embedding** | The shared linear map from flattened $P\times P$ patches to tokens; a strided convolution. | [3.2](chapters/03-vision/vit.md#what-is-it) |
| **Perplexity** | $\exp$ of the average per-token NLL. | [0.4](chapters/00-foundations/probability.md#entropy-cross-entropy-and-kl-divergence) |
| **Pixel shuffle** | Concatenating $r\times r$ neighbouring tokens into one wider token: $r^2$ fewer tokens. | [3.4](chapters/03-vision/encoders-for-vlas.md#reducing-tokens), [4.3](chapters/04-alignment/connectors.md#pixel-shuffle) |
| **Policy server** | A process hosting a model that answers observation→action-chunk requests from one or more robot clients, possibly batched. | [11.3](chapters/11-deployment/serving.md) |
| **Positional encoding** | Injection of token position: learned, sinusoidal, relative, rotary. | [2.3](chapters/02-transformers/positions.md#what-is-it) |
| **Pre-norm** | Normalising sub-layer inputs, keeping the residual path an identity. | [2.4](chapters/02-transformers/block.md#normalisation) |
| **Prefill / decode** | Parallel processing of the prompt vs. one-token-at-a-time generation. | [2.6](chapters/02-transformers/inference.md#prefill-and-decode) |
| **Prefix-LM** | Bidirectional attention over a prefix (image, prompt) and causal attention over the output. | [2.4](chapters/02-transformers/block.md#encoder-decoder-encoderdecoder), [5.3](chapters/05-vlms/small-vlms.md#paligemma-a-prefix-lm-with-location-tokens) |
| **Proprioception** | The robot's internal state: joint angles, velocities, gripper, vehicle attitude. | [Notation](appendix/notation.md#learning-and-control), [1.3](chapters/01-multimodal-intro/combining.md#actions-and-proprioception-as-modalities) |
| **QoS (ROS 2)** | Quality-of-service settings of a topic (reliability, history, durability); incompatible settings silently prevent delivery. | [11.4](chapters/11-deployment/ros2.md) |
| **Quantisation** | Storing weights (and possibly activations) with fewer bits, e.g. int8 or int4 with per-group scales. | [11.2](chapters/11-deployment/memory-quantization.md) |
| **Receding horizon** | Executing only the first action(s) of each predicted chunk and re-planning from the new observation. | [8.6](chapters/08-vla/chunking.md) |
| **Receptive field** | The input region that can influence a unit; grows with depth and stride. | [3.1](chapters/03-vision/cnn.md#receptive-fields) |
| **Register tokens** | Extra learned ViT tokens that absorb global computation and remove artifact tokens. | [3.2](chapters/03-vision/vit.md#the-cls-token-mean-pooling-and-registers) |
| **Representation collapse** | A trivial solution where an encoder maps every input to the same output. | [3.3](chapters/03-vision/self-supervised.md#what-is-it) |
| **Resampler (Perceiver, Q-Former)** | Learned queries that cross-attend to patch features to produce a fixed number of tokens. | [4.3](chapters/04-alignment/connectors.md#learned-query-resamplers) |
| **Residual stream** | The shared $T\times d$ state that sub-layers read from and add to. | [2.4](chapters/02-transformers/block.md#the-residual-stream) |
| **RNN** | Network with a hidden state updated recurrently over time. | [0.7](chapters/00-foundations/sequence-models.md#recurrent-neural-networks) |
| **RoPE** | Rotary position embedding: position-dependent rotations of queries and keys. | [2.3](chapters/02-transformers/positions.md#rotary-position-embedding-rope) |
| **Safety filter** | A layer that minimally modifies proposed commands so that known constraints (limits, zones, separation) hold. | [12.4](chapters/12-embodied/interfaces.md) |
| **Scaling laws** | Power-law relations between loss and parameters, data, compute. | [2.7](chapters/02-transformers/scaling.md#scaling-laws) |
| **Self-attention** | Attention in which queries, keys and values come from the same sequence. | [2.1](chapters/02-transformers/attention.md#what-is-it) |
| **Self-supervised learning** | Training on targets computed from the data itself (views, masked parts). | [3.3](chapters/03-vision/self-supervised.md) |
| **Shortcut learning** | Learning a spurious cue that predicts the training targets (e.g. the scene) instead of the intended one (the instruction). | [9.6](chapters/09-training/data-scaling.md) |
| **SigLIP** | CLIP-like model trained with an independent sigmoid loss per image–text pair. | [4.2](chapters/04-alignment/clip-siglip.md) |
| **SITL / HITL** | Software- / hardware-in-the-loop testing: real autopilot firmware (on a desktop / on the real board) against a simulator. | [13.6](chapters/13-uavs/testing.md) |
| **Softmax** | $e^{z_i}/\sum_je^{z_j}$; maps logits to a distribution. | [0.4](chapters/00-foundations/probability.md#softmax) |
| **SwiGLU** | Gated FFN $(\operatorname{SiLU}(\mathbf{x}W_1)\odot\mathbf{x}W_3)W_2$. | [2.4](chapters/02-transformers/block.md#the-feed-forward-network) |
| **Teacher forcing** | Training an autoregressive model on ground-truth prefixes. | [0.7](chapters/00-foundations/sequence-models.md#the-autoregressive-factorisation) |
| **Temperature** | Logit divisor $\tau$ controlling the sharpness of a softmax. | [0.4](chapters/00-foundations/probability.md#softmax) |
| **Temporal ensembling** | Executing, at each step, a weighted average of all chunk predictions covering that step (ACT). | [8.6](chapters/08-vla/chunking.md) |
| **Tensor** | Multi-dimensional array with a shape and dtype. | [0.1](chapters/00-foundations/tensors.md#what-is-it) |
| **Token / tokenizer / vocabulary** | Atomic unit of a sequence model; the string↔ids map; the set of $V$ tokens. | [0.5](chapters/00-foundations/tokens.md#what-is-it) |
| **Transformer** | Stack of blocks of attention + FFN with residuals and normalisation. | [0.8](chapters/00-foundations/first-look.md#transformers), [2.4](chapters/02-transformers/block.md) |
| **uXRCE-DDS** | The bridge that exposes PX4's internal topics as ROS 2 topics through a Micro XRCE-DDS Agent. | [13.2](chapters/13-uavs/px4-offboard.md) |
| **VAE / CVAE** | (Conditional) variational autoencoder: a latent-variable generative model trained with a reconstruction term and a KL term; ACT uses a CVAE. | [6.3](chapters/06-generative/vae.md) |
| **Visual encoder** | Network mapping an image to global or patch features. | [1.2](chapters/01-multimodal-intro/vision.md#how-does-it-work-pretrained-visual-features) |
| **Visual grounding** | Mapping a phrase to an image region: box, mask or point. | [4.4](chapters/04-alignment/grounding.md) |
| **Visual instruction tuning** | Training a VLM on instruction–response data about images, loss on the responses. | [5.2](chapters/05-vlms/training.md) |
| **ViT** | Vision Transformer: a transformer over image patches. | [0.8](chapters/00-foundations/first-look.md#vision-transformers), [3.2](chapters/03-vision/vit.md) |
| **VLA** | Vision-Language-Action model: a policy $\pi_\theta(\mathbf{A}_t\mid\mathbf{o}_t,\ell)$ built on a vision-language backbone. | [Home](index.md), [0.8](chapters/00-foundations/first-look.md#from-here-to-a-vla), Ch 8 |
| **VLM** | Vision-language model: vision encoder + projector + LLM. | [0.8](chapters/00-foundations/first-look.md#multimodal-models), [5.1](chapters/05-vlms/recipe.md) |
| **Watchdog** | Replaces commands with a safe default (hold, hover) when no fresh command arrives within a timeout. | [12.4](chapters/12-embodied/interfaces.md) |
| **Weight tying** | Sharing the input embedding and output head matrices. | [2.5](chapters/02-transformers/training.md#weight-tying) |
| **Wilson interval** | A confidence interval for a success rate that behaves well for small samples and rates near 0 or 1. | [9.5](chapters/09-training/evaluation.md) |
| **Zero-shot classification** | Classifying by comparing an image embedding with embeddings of class-name prompts. | [4.2](chapters/04-alignment/clip-siglip.md#zero-shot-classification-and-prompts) |
