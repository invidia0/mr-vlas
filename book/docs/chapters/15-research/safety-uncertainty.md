# 15.4 Safety, Uncertainty and Interpretability

A VLA's failures are hard to anticipate because its competence is statistical and its reasoning opaque. The architectures of Chapters 12–13 contain those failures; the research problems here aim to *predict* and *explain* them. When does a VLA know it does not know? Can its uncertainty be calibrated with guarantees? Can we see what it attends to and why it chose an action? Can safety constraints be part of the policy rather than only a filter around it?

## Uncertainty that means something

Runtime signals such as action-sample disagreement, token entropy and feature-space distance ([11.5](../11-deployment/monitoring.md)) are cheap but uncalibrated.

- **What exists.** Conformal prediction gives distribution-free coverage guarantees from a calibration set; KnowNo (Ren et al., 2023) applied it to LLM planners, so that a robot asks for help when the prediction set contains more than one option, with a statistical guarantee on task success. Ensembles and test-time sampling give empirical uncertainty for generative heads.
- **Research questions.**
    1. Can conformal methods be extended from discrete plan choices to continuous action chunks — for example, to decide when a flow-matching VLA's samples disagree enough to ask for help — with a guarantee on the rate of unsafe actions?
    2. Which uncertainty signal best predicts *task* failure (not only action error) for VLAs, and how does that change under distribution shift?
    3. For a team, can uncertainty be used to decide *which robot* asks the human for help ([14.6](../14-multi-robot/data.md), Fleet-DAgger)?

## Safety inside the policy

Safety filters ([12.4](../12-embodied/interfaces.md)) keep known constraints. A policy that constantly pushes against them is unsafe in a different way (it will find the gaps) and inefficient (it fights its own envelope).

- **Research questions.**
    1. Can filter interventions be used as a training signal (a penalty in RL fine-tuning, or negative examples) so that the policy learns to propose safe actions, while keeping the filter as the guarantee?
    2. How should a VLA represent constraints it cannot see — geofences and no-fly zones from a map — as input tokens, and does that reduce interventions without making the policy dependent on them?
    3. What is the right "safe fallback" behaviour for each embodiment and task, and can it be learned from data rather than hand-specified?

## Interpretability

- **What exists.** Attention maps and the logit lens for transformers ([Lab 05](../02-transformers/inference.md)), probing representations ([Lab 07](../04-alignment/contrastive.md)), and grounding outputs (boxes, points) that make a VLM's decisions inspectable ([5.4](../05-vlms/spatial.md)).
- **Research questions.**
    1. When a VLA picks the wrong object, is the error in grounding (the vision-language part) or in control (the action head)? Design probes that separate the two, e.g. by decoding the target location from intermediate features.
    2. Do VLAs contain identifiable internal features for the instructed target, and can steering them change behaviour predictably — a tool for debugging and for safety audits?
    3. Can a VLA be trained to produce a short, checkable explanation (the target it chose, as a box) alongside each action chunk, at negligible latency, and does checking that explanation catch errors?

## A first experiment with the course tools

In Lab 11, compute action-sample disagreement for the flow head on ambiguous vs specific instructions ([11.5](../11-deployment/monitoring.md)'s code). Calibrate a threshold on held-out episodes so that "ask for help" is triggered on at most 10% of episodes, and measure how many failures it catches. Then train a linear probe that decodes the target object's position from the backbone's pooled features and check whether probe errors predict the wrong-object failures of Lab 11's failure analysis.
