# LightLOB W4A4 on PYNQ-Z2 (Zynq-7020)

Model: LightLOB, 4-bit weights and activations (first/last layer 8-bit weights), FI-2010, k = 10.
Build: FINN, Vivado 2024.2, `target_fps = 6000`, `synth_clk_period_ns = 14` (71.4 MHz target),
`mvau_wwidth_max = 10000`.

| Item | Value |
|---|---|
| Timing | WNS −0.183 ns at 14 ns → closes at about 70.5 MHz; run at 66.67 MHz (1000/15) |
| Correctness | 1,998 / 1,998 windows identical to PyTorch argmax, max abs logit difference 4.8e-7 |
| Accuracy on those windows | 0.5731 |
| Batch-1 accelerator latency | median 0.563 ms, p99 0.652 ms, min 0.547 ms (200 runs) |
| End-to-end (Python driver, ARM-side I/O) | about 1.915 ms |
| Throughput (batch 500) | 5,697 inferences/s |
| FINN estimate | 6,132 fps; critical path 203,959 cycles (latency estimate ≈ 5× too pessimistic) |

An earlier 100 MHz build (`target_fps = 10000`) used 33,621 LUTs post-synthesis but failed timing
(WNS −3.524 ns); a 14 ns build at `target_fps = 10000` failed placement.

## CPU / GPU baselines (batch 1, `measure_latency.py`, shared lab server, RTX 4090)

| Model | CPU 1 thread median / p99 | GPU median / p99 |
|---|---|---|
| DeepLOB | 3,510 µs / 10,180 µs | 1,110 µs / 18,054 µs |
| LightLOB | 455 µs / 13,250 µs | 553 µs / 16,670 µs |

GPU timings include host→device copy and synchronisation; the p99 values include contention from
other jobs on the shared server.

## LightLOB2 FINN estimates (14 ns clock, not built)

| Variant | Estimated throughput | LUT |
|---|---|---|
| LightLOB (v1) W2A4 | capped at 17,848 fps (input bottleneck) | – |
| LightLOB2 W2A4 | 6,131 fps | 15.5k |
| LightLOB2 W2A4 | 35,714 fps | 39.5k |
| LightLOB2 W2A4 | 101,461 fps | 122k (larger device) |

LightLOB2 W4A4 needs about 40% more LUTs than W2A4.
