# Chapter 13 Review

## Key ideas

1. The **flight controller** owns the vehicle: estimator, cascaded controllers at 50 Hz–8 kHz, geofence and failsafes. The **companion computer**, and any VLA on it or on the ground, can only request behaviour.
2. **PX4 offboard mode** tracks streamed setpoints while a heartbeat (≥ 2 Hz) continues; loss of the stream triggers a failsafe. uXRCE-DDS exposes PX4 topics in ROS 2 (best-effort QoS); NED vs ENU frames and NaN-for-unused fields are the classic first-flight bugs.
3. Of four placements — mission writer, **goal/waypoint proposer**, trajectory/chunk producer, direct velocity streamer — the goal level gives the model the semantic decisions and keeps flight in classical hands; latency then delays decisions instead of destabilising flight.
4. **Direct action** is risky: latency is dead time ($k\delta>\pi/2$ is unstable), unusual states are absent from demonstrations, failures are unrecoverable, and silence leaves a stale command in force. It needs a full safety envelope, if used at all.
5. **Defence in depth**: goal validation, planning with margins, CBF/limit filtering with margins for lag and wind, a companion watchdog, PX4's independent failsafes, and a human override that depends on none of them; authority defined for every situation.
6. **Test ladder**: SITL → HITL → bench → tethered → field, with failure injection at every stage, shadow mode for perception, and complete logs.

## Exercises

1. **Stack budget.** For a 1.5 kg quadrotor with a 75 Wh battery, estimate hover time with no companion, with a 15 W board and with a 40 W board (using [13.1](stack.md)'s rough model). Then estimate the latency of a ground-hosted SmolVLA over a 30 ms link, and decide where you would run it for a 10-minute inspection mission.
2. **Frames.** A VLA trained in ENU outputs the velocity $(1,0,0.2)$ m/s. Write the NED setpoint PX4 must receive. What does the drone do if you forget the conversion?
3. **Dead time.** For $\dot x=k(x^\ast-x(t-\delta))$, find the largest $k$ that is stable for $\delta=0.3$ s, and the largest $k$ that does not overshoot. Compare with the toy VLA's behaviour in Lab 15.
4. **Authority table.** Extend the table of [13.5](safety.md) to a two-drone mission with a shared ground server. Who is in authority when the server fails? When one drone's link fails?
5. **Failure injection.** Write expected outcomes for three new failures: a GNSS glitch that shifts the position estimate by 5 m, a VLA that confidently outputs a goal behind a wall missing from the map, and a companion computer that thermally throttles to half speed.
6. **Research question.** Design an experiment that measures, in SITL, how the probability of a geofence breach depends on VLA latency for architectures B and D, with enough episodes to give 95% intervals on rates below 1% ([9.5](../09-training/evaluation.md)).

<!-- quiz: ch13 -->

## Reading assignments

1. PX4 User Guide — [Offboard mode](https://docs.px4.io/main/en/flight_modes/offboard.html) and [ROS 2 user guide](https://docs.px4.io/main/en/ros2/user_guide.html) (uXRCE-DDS, `px4_msgs`, frames).
2. PX4 User Guide — [Safety (failsafes, geofence)](https://docs.px4.io/main/en/config/safety.html).
3. Ames et al. (2019), *Control Barrier Functions: Theory and Applications*, [arXiv:1903.11199](https://arxiv.org/abs/1903.11199).
4. Kaufmann et al. (2023), *Champion-level drone racing using deep reinforcement learning*, [Nature 620, 982–987](https://www.nature.com/articles/s41586-023-06419-4) — what direct learned control of a drone takes: small fast policies, simulation, and a tightly controlled environment.
