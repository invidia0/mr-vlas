# Chapter 1 Review

## Key ideas

1. A language model is used through a **generation loop**: forward pass → last-position logits → **decoding rule** → append. The decoding rule becomes part of the controller when the outputs are actions.
2. The **context window** is the model's only memory.
3. An image is a tensor $C\times H\times W$; pixels are a poor representation; pretrained **visual encoders** produce global or **patch features**, whose properties depend on the pretraining signal (labels, text contrast, self-supervision).
4. Robot images differ from internet images (viewpoint, motion, geometry, multiple cameras).
5. **Fusion**: late vs early vs intermediate; mechanisms are in-sequence concatenation, cross-attention, FiLM and adaptive normalisation, with different interaction/cost trade-offs.
6. **Tokens as a universal interface** won because it reuses pretrained LLMs unmodified.
7. **Alignment is not grounding**: connecting words to actions needs robot data.

## Exercises

1. **Decoding.** Implement top-$k$ sampling and prove that top-$k$ with $k=1$ equals greedy decoding. When does top-$p$ with $p\to0$ equal greedy?
2. **Context budget.** A VLM has a 4096-token context. Each camera image is 256 tokens; the system prompt is 150 tokens; each robot status message from teammates is 40 tokens. How many past frames of two cameras can a drone keep in context if it must also read reports from 5 teammates?
3. **FiLM as attention.** Show that FiLM with $\boldsymbol\beta=0$ is a special case of a gating mechanism, and that it cannot express "attend to the patch containing the red object" — i.e. it cannot make the modulation depend on spatial position. Which mechanism can?
4. **Design.** For the drone's survey task, sketch a late-fusion pipeline with a CLIP-style encoder that flags frames likely to contain the red toolbox. Estimate per-frame cost in tokens and compare with running a full VLM on every frame.
5. **Grounding.** List three facts about the red toolbox that are needed to *bring it to the inspection station* and that cannot be learned from image–caption pairs alone.

<!-- quiz: ch01 -->

## Reading assignments

1. [Radford et al. (2019), GPT-2](../../papers.md#radford2019) — Section 2: tasks as text. The same idea lets an instruction specify a robot's task.
2. [Brown et al. (2020), GPT-3](../../papers.md#brown2020) — the in-context-learning framing (Figure 1.2).
