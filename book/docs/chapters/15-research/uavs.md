# 15.2 VLAs for Aerial Robots

Vision-language-action research on drones is young. Early systems split along the architectural lines of [13.3](../13-uavs/architectures.md). Some generate missions from language and satellite imagery. Others fine-tune VLAs to output velocity commands from first-person video in simulation, with or without a separate reasoning model. This section reviews them briefly and lists the problems between them and safe, useful aerial VLAs.

## Where the field is

| System | What it does | Architecture ([13.3](../13-uavs/architectures.md)) |
|---|---|---|
| UAV-VLA (Sautenkov et al., 2025) | generates flight paths and action plans for large-scale missions from a text request, combining a VLM on satellite imagery with an LLM | A: mission writer |
| RaceVLA (Serpiva et al., 2025) | a VLA fine-tuned on racing-drone data for navigation from a first-person camera, compared with OpenVLA and RT-2 on generalisation axes | D/C: direct commands |
| CognitiveDrone (Lykov et al., 2025) | a VLA producing 4-D action commands from first-person images and instructions for tasks requiring recognition, symbol understanding and reasoning; trained on ~8 k simulated trajectories; adding a VLM reasoning module that simplifies the instruction first (CognitiveDrone-R1) raised benchmark success from 59.6% to 77.2% | D with a slow reasoning layer (dual-system, cf. [10.5](../10-case-studies/groot-smolvla.md)) |

Two patterns stand out. Separating slow reasoning from fast control helps, as in GR00T N1 and π0.5 for manipulation. And most results are in simulation, with the safety architecture outside the scope of the papers.

## Open problems

### 1. Goal-level aerial VLAs

Architecture B — the VLA proposes goals, a planner flies ([13.3](../13-uavs/architectures.md)) — is the safest place for a VLA on a drone, yet there are few datasets and models for it.

- **Research questions.**
    1. How should aerial demonstrations be labelled with goals (waypoints, regions, object references)? Can goals be extracted automatically from flight logs (e.g. hover points, the end of each chunk as in Lab 15)?
    2. Does a goal-level VLA trained in simulation transfer better to real flight than a direct-action VLA, because the classical layers absorb the dynamics gap?
    3. What goal representation (3-D point, image-space box plus depth, map cell, language reference) gives the best grounding accuracy from altitude?

### 2. Perception from altitude

VLMs are trained mostly on ground-level photos. Aerial and oblique views, small objects, motion blur and changing altitude are out of distribution ([5.4](../05-vlms/spatial.md)).

- **Research questions.**
    1. How does grounding accuracy of small VLMs degrade with altitude and viewing angle, and how much aerial fine-tuning data recovers it?
    2. Can a drone actively choose viewpoints (descend, orbit) to reduce a VLM's uncertainty before committing to a goal?

### 3. Latency-aware aerial control

Lab 15 showed that latency turns direct control unstable while goal-level control degrades gracefully. Real-time chunking methods for manipulation (e.g. RTC, Black et al., 2025: computing the next chunk while executing the current one, freezing the actions that will execute and inpainting the rest) address the chunk-boundary problem without retraining.

- **Research questions.**
    1. Do real-time chunking methods make chunked aerial VLAs (architecture C) safe enough at realistic companion-computer latencies, measured as geofence-breach rates in SITL ([13.6](../13-uavs/testing.md))?
    2. Can speed limits that adapt to measured latency ([13.5](../13-uavs/safety.md)) be derived formally from the vehicle's dynamics and the delay distribution?

### 4. Safety evidence for learned aerial autonomy

- **Research questions.**
    1. How many SITL hours, with which failure injections, are needed to bound the probability of a geofence breach below a target for a VLA-driven drone ([9.5](../09-training/evaluation.md))?
    2. Which runtime monitors (action disagreement, OOD scores, filter intervention rate) best predict unsafe episodes before they happen?

## A first experiment with the course tools

In Lab 15, replace the waypoint extraction (chunk endpoint) by a VLA trained to predict the *target position* directly (a 2-D goal head). Compare goal accuracy, success and violation rates under latency and wind with the direct and chunk-endpoint variants. Then port the comparison to PX4 SITL with the offboard node of [Appendix B](../../appendix/ros2-px4.md).
