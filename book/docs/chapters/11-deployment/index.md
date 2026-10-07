# 11. Inference, Deployment and Systems Engineering

A VLA that works in a paper is a model; a VLA that works on a robot is a system. This chapter covers the engineering between the two: latency budgets measured at the tail, memory and quantisation for edge computers, serving several robots from one GPU with asynchronous execution, integration as a well-behaved ROS 2 component, and the monitoring, logging and versioning that let you find and reproduce failures. It turns the cost models of Chapters 2, 6 and 8 into procedures.

By the end you should be able to take a trained VLA and decide where it runs, at what precision, at what rate, how it talks to the rest of the robot, what happens when it is late, and what you will log to debug it.

## Learning objectives

1. Write and measure a latency budget at the p99, and choose among the levers that close it. ([11.1](latency.md))
2. Compute memory at each precision, explain weight-only quantisation methods and their effect on control, and match models to edge hardware by bandwidth. ([11.2](memory-quantization.md))
3. Choose a serving placement, batch requests from several robots, and implement asynchronous chunk execution over a network. ([11.3](serving.md))
4. Design a ROS 2 VLA node: topics, QoS, threading, timestamps, and a separate safety node with authority over commands. ([11.4](ros2.md))
5. Log what is needed to replay failures, monitor uncertainty and out-of-distribution signals, and version a deployment for reproducibility. ([11.5](monitoring.md))

## Sections

| Section | Topic | Lab |
|---|---|---|
| [11.1](latency.md) | Latency budgets and control rates | Lab 14 |
| [11.2](memory-quantization.md) | Memory, quantisation and edge hardware | Lab 14 |
| [11.3](serving.md) | Serving, batching and asynchronous execution | Lab 14 |
| [11.4](ros2.md) | ROS 2 integration | Appendix B |
| [11.5](monitoring.md) | Monitoring, logging, failure modes, reproducibility | Labs 11, 14 |
| [Review](review.md) | Quiz and reading assignments | |

## Labs

- **Lab 14 — Inside SmolVLA** (`labs/14_smolvla/`) covers deployment measurements: stage-by-stage latency with percentiles, quantisation, and a simulated multi-robot policy server.
