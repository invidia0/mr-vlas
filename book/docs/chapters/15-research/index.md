# 15. Research Directions

This last chapter is a map of open problems at the intersection of VLAs, multi-robot systems and aerial robotics. For each area it says what exists, why the problem is hard, and which concrete research questions could become papers. Each section ends with a first experiment you can run with the course's code. The chapter assumes Chapters 8–14: the questions are phrased in their vocabulary and point back to the sections and labs where the necessary tools were built.

## Sections

| Section | Area | Builds on |
|---|---|---|
| [15.1](multi-robot.md) | VLAs for multi-robot systems; decentralised foundation models | Ch. 14, Lab 16 |
| [15.2](uavs.md) | VLAs for aerial robots | Ch. 13, Lab 15 |
| [15.3](generalisation.md) | Long-horizon autonomy, generalisation, data efficiency | Ch. 9, Labs 11–13 |
| [15.4](safety-uncertainty.md) | Safety, uncertainty, interpretability | Ch. 11–12 |
| [15.5](realtime-communication.md) | Communication-aware and real-time policies | Ch. 8, 11, 14; Labs 15–16 |
| [15.6](methodology.md) | From research question to paper | Ch. 9.5; all labs |
| [Review](review.md) | Quiz, exercises, reading | |

## Five themes that run through the chapter

1. **Put the foundation model where its strengths are.** Semantics, grounding and task structure from VLMs and VLAs; stability, optimisation and guarantees from classical layers.
2. **Treat time and communication as part of the problem**, not as deployment details.
3. **Teams need team data and team evaluation**; single-robot competence does not compose into coordination.
4. **Uncertainty should be calibrated and actionable**: asking for help, slowing down, or deferring to a teammate.
5. **Evidence over anecdotes**: paired evaluation, intervals, strong simple baselines, stress tests.
