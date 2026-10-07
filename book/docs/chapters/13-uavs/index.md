# 13. VLAs for UAVs

Drones are where the gap between what a vision-language model can decide and what a vehicle can safely be told is widest. They benefit enormously from semantic understanding ("inspect the damaged panel", "follow the red truck", "find the person in the orange jacket"). Yet their dynamics are fast, their failures unrecoverable, and their onboard compute limited. This chapter builds the UAV-specific engineering on top of Chapters 11 and 12: the UAV stack and PX4's offboard interface, the architectures for placing a VLA in it, why direct action prediction is risky, a defence-in-depth safety architecture, and the testing ladder from simulation to field.

The position taken throughout: **the VLA proposes goals, classical layers fly**. Any extra authority given to the model must come with the protections that make it safe to grant.

## Learning objectives

1. Describe the UAV stack — flight controller loops and failsafes, companion computer, links — and the rates and responsibilities of each. ([13.1](stack.md))
2. Use PX4 offboard control over uXRCE-DDS from ROS 2: the heartbeat protocol, setpoint types, frames, QoS and failsafes. ([13.2](px4-offboard.md))
3. Compare four placements of a vision-language model in a UAV stack and justify the goal-level architecture. ([13.3](architectures.md))
4. Explain the risks of direct action — latency as dead time, distribution shift, unrecoverable failures, silence — and quantify the first. ([13.4](direct-action.md))
5. Specify a defence-in-depth safety architecture with explicit authority, and implement its companion-side layers. ([13.5](safety.md))
6. Plan a SITL → HITL → bench → tethered → field test campaign with failure injection and logging. ([13.6](testing.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [13.1](stack.md) | The UAV stack | |
| [13.2](px4-offboard.md) | PX4, offboard control and uXRCE-DDS | Appendix B |
| [13.3](architectures.md) | VLA → planner → trajectory → controller → PX4 | Lab 15 |
| [13.4](direct-action.md) | Direct action prediction and why it is risky | Lab 15 |
| [13.5](safety.md) | Safety architecture | Lab 15 |
| [13.6](testing.md) | SITL, HITL, tethered and field testing | Appendix B |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 15 — Flying a VLA safely** (`labs/15_uav_safety/`): the toy VLA of Lab 11 flies a lagged, wind-blown drone over the depot, either streaming its actions directly or proposing waypoints for a classical controller; keep-out zones it cannot see, a CBF filter with margins, a watchdog, latency up to 0.5 s and link loss.
- **Appendix B** sets up ROS 2, PX4 SITL and Gazebo, and runs the offboard node and safety supervisor of this chapter in simulation.

!!! warning "Before any real flight"
    This chapter teaches architecture and engineering; it is not a substitute for your institution's flight-safety procedures, a qualified safety pilot, or the aviation regulations where you fly.
