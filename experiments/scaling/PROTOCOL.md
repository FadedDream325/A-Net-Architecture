# A-Net Scaling and Cross-Layer Validation Protocol

Status: experiment branch experiment/scaling-and-crosslayer.
This file documents the design before interpreting the full results.

## 1. Research questions

1. Does the current A-Net (shared L1 + task experts + Task-ID Top-1) preserve
   its accuracy/compute trade-off as width increases from roughly 100k to 5M
   A-Net parameters?
2. Does A-Net reduce unique parameters, checkpoint size, active parameters,
   MACs, latency or training memory relative to an independent model and a
   shared-trunk model?
3. Does a direct cross-layer parameter-block call add measurable value after
   correctness tests, or is it only extra parameters/compute?
4. Are any gains stable across three seeds and three task difficulties?

## 2. Implementation status (do not conflate)

| Concept | Status in this repository |
|---|---|
| A. Shared L1, task-specific experts, known Task-ID Top-1 routing | Implemented in multitask_anet_demo.py; measured here |
| B. One workflow directly calling another layer's parameter block | Minimal fixed-call-graph prototype implemented in experiments/scaling/models.py (ANetCrossLayer); mechanism tests only, not a general cross-layer system |
| C. Learned parameter sharing or learned routing | Not implemented |

The A-Net measurements below are evidence for A only. They are not evidence
for B or C. B is evaluated separately as a small ablation, not assumed to work.

## 3. Task suite

All tasks are 2D binary classification with a controlled train/test split.

| id | task | difficulty | family | train noise |
|---|---|---|---|---|
| T1 | Easy Sine (x2 > 0.8*sin(2*x1)) | easy | sine | 1% |
| T2 | Medium Sine (x2 > 0.55*sin(3*x1) + 0.28*sin(7.5*x1)) | medium | sine | 2% |
| T3 | Hard Checker (25-degree rotated, 1.1-wide checkerboard) | hard | checkerboard | 2% |

- T1 and T2 are correlated (same sine family); T3 is uncorrelated.
- 12,000 training / 3,000 test samples per task.
- Label noise is injected into training labels only; test labels are clean.
- The same tensors are used for every architecture at a given seed.

## 4. Model scales

Width is identical across architectures within a scale. A-Net's total parameter
count is the scale anchor. Parameter counts are reported rather than forced to
match, because the architectures have inherently different sharing patterns.

| scale | hidden | A-Net expert | A-Net target | Independent total | Shared total | A-Net total |
|---|---:|---:|---:|---:|---:|---:|
| small | 256 | 128 | ~100k | 201,222 | 68,102 | 100,230 |
| medium | 896 | 448 | ~1.2M | 2,424,582 | 811,782 | 1,210,950 |
| large | 1792 | 896 | ~4.8M | 9,666,054 | 3,229,190 | 4,830,342 |

The cross-layer prototype adds an expert-hidden-sized shared block:

- small: 116,742 total parameters
- medium: 1,412,102 total parameters
- large: 5,634,054 total parameters

## 5. Training protocol

- Optimizer: AdamW, lr = 1e-3, weight decay = 1e-4.
- 50 epochs, batch size 512, FP32.
- Three seeds: 2026, 2027, 2028.
- Data preloaded to the GPU; training time excludes I/O and preprocessing.
- CUDA/cuBLAS and every architecture are warmed up before the measured runs,
  so training times exclude CUDA context initialization.
- Same model-initialization seed for every architecture at a given seed.
- Same task tensors and evaluation protocol for all architectures.
- No early stopping, no seed removal, no test-set selection.

## 6. Metrics and accounting

- Unique trainable parameters: sum of all trainable parameter elements,
  including experts that are not executed by a given sample.
- Active parameters per sample: parameters actually used in one forward
  pass. Independent = one MLP; Shared = entire trunk+one head; A-Net = L1 +
  one expert; Cross = L1 + one expert + one shared block.
- MACs / FLOPs: analytical forward count for one sample, ignoring biases
  and activation functions:
  - Independent / Shared: h^2 + 4h MACs.
  - A-Net: 2h + h*e + 2e MACs.
  - Cross: A-Net + e^2 MACs.
  - FLOPs = 2 × MACs.
  - The formulas are checked against torch.utils.flop_counter in
    tests/test_scaling_models.py.
  - Routing is Python integer indexing and contributes no tensor MACs.
- Checkpoint size: measured from the actual saved state dict.
- Training peak memory: torch.cuda.max_memory_allocated during the run.
- Inference latency: CUDA Event P50/P95, batch=1, after training.
  first_call_after_training_us is reported separately; it is not a true
  process cold start. A separate fresh-process benchmark is provided for true
  cold start on one seed.
- Throughput: batch=1024 samples/second, measured with wall time around
  repeated forwards.
- Routing distribution: Task ID deterministically selects one branch; the
  runner records the selected branch for each task. Correctness tests verify
  only that branch receives gradients.
- Failures: every run is written as JSONL with status ok or failed;
  non-finite losses/gradients raise and are recorded.

## 7. Fairness controls

- Identical data, splits, seeds, optimizer, epochs, batch size and precision.
- Identical width within a scale.
- No architecture-specific hyperparameter tuning was performed after seeing
  test results.
- Three seeds are required for aggregation; no seed is dropped.
- The independent model has more total parameters by design; this is reported
  rather than hidden. The shared model has fewer total parameters but all are
  active. A-Net's total/active distinction is explicit.

## 8. Cross-layer prototype specification

ANetCrossLayer in experiments/scaling/models.py:

    x -> shared L1 -> expert[t].fc1 -> shared_block -> expert[t].fc2

- shared_block is registered once on the parent model.
- It is passed as a module argument into the selected expert; the expert does
  not own or copy it.
- Input/output shape is (batch, expert_hidden_dim).
- The block is a Linear(e, e) + ReLU residual call.
- use_cross_layer=False skips the call but leaves parameters registered,
  providing a same-parameter-count compute ablation.
- Tests verify: one registration, one block object, direct numeric equivalence,
  gradient accumulation from two task paths, only selected expert gradients,
  and zero block calls when disabled.
- Post-hoc compute ablation: cross_off uses the same total parameter count but
  skips the block call; it is run for all three scales and three seeds.

## 9. Commands

Preflight:

~~~bash
python -m experiments.scaling.run_scaling \
  --scales small --models independent shared anet cross \
  --seeds 2026 --epochs 2 --max-steps 30 \
  --out-dir experiments/scaling/results/preflight
~~~

Full run:

~~~bash
python -m experiments.scaling.run_scaling \
  --scales small medium large \
  --models independent shared anet cross \
  --seeds 2026 2027 2028 \
  --epochs 50 \
  --out-dir experiments/scaling/results/scaling_v1 \
  --resume
~~~

Analysis:

~~~bash
python -m experiments.scaling.analyze_results \
  --results-dir experiments/scaling/results/scaling_v1
~~~

True cold start (fresh process per checkpoint, seed 2026):

~~~bash
python -m experiments.scaling.benchmark_cold_starts \
  --results-dir experiments/scaling/results/scaling_v1 --seed 2026
~~~

Tests:

~~~bash
python -m pytest -q -p no:cacheprovider tests
~~~

## 10. Known limitations

- Only synthetic 2D tasks; results do not establish behavior on vision,
  language or tabular benchmarks.
- Three seeds are the minimum planned; confidence intervals are correspondingly
  coarse.
- The cross-layer prototype is a fixed call graph, not learned or general
  cross-layer routing.
- GPU is shared with desktop applications; latency is measured but may vary.
- Only one GPU and one PyTorch/CUDA build were tested.
- No TensorRT/CUDA Graph/ONNX static deployment is included.
