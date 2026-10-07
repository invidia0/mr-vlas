# Chapter 11 Review

## Key ideas

1. A **latency budget** covers sensing, transport, preprocessing, model, post-processing and hand-off, at the **p99**, against a deadline set by the chunk execution horizon or the tolerable staleness. Decide in advance what happens when the deadline is missed.
2. **Bytes per weight** decide what fits and how fast bandwidth-bound phases run. Weight-only int8/int4 quantisation (RTN, NF4, GPTQ, AWQ) cuts memory 2–4×; speed depends on kernels; validate quantised policies in closed loop on precision tasks. Match models to edge devices by bandwidth.
3. **Serving**: on board, edge server or cloud; a policy server can **batch** requests from a team because single-request decoding leaves the GPU idle; **asynchronous** execution with time-aligned chunks hides network and inference delay at the cost of staleness.
4. **ROS 2**: the VLA node subscribes with sensor-data QoS, keeps inference off the control path, publishes **time-stamped chunks**, and never commands actuators directly — a safety node does.
5. **Monitoring and reproducibility**: log inputs, prompts, chunks, filtered commands, latencies and model identity; use sample disagreement, token entropy, feature-space OOD scores and progress stalls as (uncalibrated) warning signals; version weights + normalisation + prompt + preprocessing together.

## Exercises

1. **Budget.** A two-camera arm VLA: camera 15 ms, transport 5 ms, preprocessing 8 ms, vision 2×12 ms, prefill 70 ms, expert 10×3 ms, hand-off 2 ms, with a p99 inflation of 1.6× on the model stages. At 20 Hz control, what is the smallest synchronous $H_e$? With asynchronous execution, what is the staleness of the first new action?
2. **Quantisation.** For a 3 B VLA on an Orin NX (16 GB shared, 102 GB/s), compute the decode-step lower bound at bf16, int8 and int4, and what remains of the memory for other processes in each case.
3. **Batching.** Ten robots at 10 Hz with chunks of 10 share one GPU. The model's batch-1 prefill takes 40 ms and batch-10 takes 70 ms. Compare one-by-one serving with batching every 20 ms: utilisation, per-robot staleness, and whether deadlines are met.
4. **QoS bug.** A camera publisher uses `BEST_EFFORT`, the VLA subscriber `RELIABLE`. What happens, and how do you detect it? What happens with the opposite combination over Wi-Fi?
5. **Monitoring.** Propose three runtime signals for a drone that uses a VLM to confirm "red toolbox found", with a response for each, and explain how you would set their thresholds.
6. **Research question.** A fleet shares one policy server over Wi-Fi. Design a scheduling policy that minimises the probability that any robot runs out of actions, using each robot's queue length and network latency estimate. How does it behave when one robot's link degrades?

<!-- quiz: ch11 -->

## Reading assignments

1. [Kim et al. (2024), OpenVLA](../../papers.md#kim2024) — the quantisation and inference-speed results.
2. [Shukor et al. (2025), SmolVLA](../../papers.md#shukor2025) — asynchronous inference.
3. Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — the NF4 data type.
4. Lin et al. (2024), *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) — Sections 3–4.
5. Macenski et al. (2022), *Robot Operating System 2: Design, Architecture, and Uses in the Wild*, Science Robotics, [arXiv:2211.07752](https://arxiv.org/abs/2211.07752) — the design rationale of ROS 2.
