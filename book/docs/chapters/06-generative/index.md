# 6. Generative Models for Continuous Outputs

Part III builds the half of a VLA that acts. It starts with a statistical fact rather than an architecture. Demonstrations of the same situation contain different valid actions, and a network trained to regress them predicts their average, which can be invalid. This chapter builds the models that represent whole action distributions instead: mixtures, discretised autoregressive tokens, variational autoencoders, diffusion and flow matching. Each comes with its mathematics, a shape-annotated implementation, and a measurement of its failure modes and sampling cost on one shared benchmark (Lab 09).

By the end you should be able to derive the ELBO and the diffusion loss, implement flow matching in a few lines, explain why each model does or does not average modes, and estimate what each one costs inside a control loop.

## Learning objectives

1. Prove that squared-error training predicts the conditional mean, and list where multimodality comes from in robot data. ([6.1](multimodality.md))
2. Implement mixture density networks and discretised autoregressive action heads, and explain why the second dimension must condition on the first. ([6.2](mdn-ar.md))
3. Derive the ELBO, the reparameterisation trick and the KL term; explain posterior collapse and how ACT uses a CVAE. ([6.3](vae.md))
4. Derive DDPM's training loss from the ELBO, relate noise prediction to the score, and explain DDIM. ([6.4](diffusion.md))
5. Derive conditional flow matching with linear paths and explain why regressing averaged velocities still produces a multimodal distribution. ([6.5](flow-matching.md))
6. Build a latency model for each generator and relate it to control rate, action chunking and asynchronous execution. ([6.6](sampling-cost.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [6.1](multimodality.md) | Regression to the mean and multimodal action distributions | Lab 09 |
| [6.2](mdn-ar.md) | Mixtures and discretisation | Lab 09 |
| [6.3](vae.md) | Variational autoencoders and the CVAE of ACT | Lab 09 |
| [6.4](diffusion.md) | Denoising diffusion | Lab 09 |
| [6.5](flow-matching.md) | Flow matching and rectified flow | Lab 09 |
| [6.6](sampling-cost.md) | Sampling cost and control rates | Lab 09 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 09 — Generative action heads** (`labs/09_generative_actions/`): one bimodal "pass the obstacle" task. MSE regression, MDNs, independent and autoregressive bins, CVAEs at three β, DDPM/DDIM and flow matching are each trained on it and compared on invalid samples, mode balance, network evaluations and microseconds per sample.
