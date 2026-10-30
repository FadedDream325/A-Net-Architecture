# A-Net

**Shared shallow interface + task-specific experts + Task-ID Top-1 hard routing**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-research%20prototype-6B7280)]()

> 这是一个探索性的小型多任务学习项目，不是完整大模型的结论。实验受限于单张 RTX 5060 Laptop 和合成任务，欢迎指正。

## What is implemented

A-Net currently means:

1. a shared shallow layer L1;
2. one nonlinear expert network per task;
3. deterministic Top-1 routing by known Task ID.

Only the selected expert is executed. Unselected experts do not receive
gradients. The route is **not learned**, and there is no automatic task
discovery.

~~~text
x -> Shared L1 -> expert[task_id] -> logits
                    ^
                    only this expert runs
~~~

## Implementation status

| Concept | Status |
|---|---|
| A. Shared L1 + task experts + known Task-ID Top-1 routing | Implemented in multitask_anet_demo.py and measured in the new scaling study |
| B. A workflow directly calling another layer's parameter block | Minimal fixed-call-graph prototype in experiments/scaling/models.py; correctness tested, no demonstrated general gain |
| C. Learned parameter sharing or learned routing | Not implemented |

A-Net results below are evidence for A only. Do not read them as validation of
B or C.

## New controlled scaling results

Protocol: 3 synthetic 2D tasks (easy sine / medium sine / hard rotated
checkerboard), 12,000 train / 3,000 test each, 50 epochs, 3 seeds, same width
for every architecture, RTX 5060 Laptop, PyTorch 2.14.1+cu130.

| scale | model | total params | active params | mean accuracy (3 seeds) |
|---|---|---:|---:|---:|
| small | Independent | 201,222 | 67,074 | 98.29 ± 0.05 |
| small | Shared trunk + head | 68,102 | 68,102 | 97.67 ± 0.34 |
| small | A-Net | 100,230 | 33,922 | **98.09 ± 0.04** |
| small | A-Net + cross block | 116,742 | 50,434 | 98.02 ± 0.47 |
| medium | Independent | 2,424,582 | 808,194 | 98.25 ± 0.25 |
| medium | Shared trunk + head | 811,782 | 811,782 | 97.45 ± 0.79 |
| medium | A-Net | 1,210,950 | 405,442 | **98.10 ± 0.21** |
| medium | A-Net + cross block | 1,412,102 | 606,594 | **98.41 ± 0.11** |
| large | Independent | 9,666,054 | 3,222,018 | 98.20 ± 0.12 |
| large | Shared trunk + head | 3,229,190 | 3,229,190 | 97.47 ± 0.10 |
| large | A-Net | 4,830,342 | 1,613,698 | **98.24 ± 0.20** |
| large | A-Net + cross block | 5,634,054 | 2,417,410 | 98.24 ± 0.02 |

Key observations:

- A-Net uses about **50% of the independent baseline's total parameters,
  active parameters and checkpoint size** at every scale.
- A-Net's average accuracy is -0.20 / -0.15 / +0.04 pp versus independent,
  i.e. essentially parity within seed noise.
- A-Net is **not a universal latency win**: batch=1 latency is 1.30–1.49×
  slower than the independent baseline. At batch=1024 it reaches 0.69× /
  1.25× / 1.64× the independent throughput for small / medium / large.
- A-Net beats the shared trunk by +0.41 / +0.65 / +0.77 pp while using half
  of its active compute, but A-Net stores about 1.5× more parameters than the
  shared model.
- The fixed cross-layer block is not a consistent win: it adds ~17% total
  parameters and ~50% active compute; accuracy vs A-Net is -0.07 / +0.31 /
  -0.00 pp. See the full report.
- True process-level cold start is 8–11 s per checkpoint in this environment,
  dominated by the first GPU forward; the first model forward takes 6–9 s.

![Accuracy vs total parameters](experiments/scaling/results/scaling_v1/figures/accuracy_vs_total_params.png)

Full report: [experiments/scaling/REPORT.md](experiments/scaling/REPORT.md)

Protocol: [experiments/scaling/PROTOCOL.md](experiments/scaling/PROTOCOL.md)

Raw data: [experiments/scaling/results/scaling_v1](experiments/scaling/results/scaling_v1)

## Historical result correction

The older README claimed A-Net activated 17% of the independent baseline's
parameters. That number was computed as 8,770 / 51,462 with an A-Net expert
width of 64 versus an independent hidden width of 128. It is a real number for
that specific configuration, but it mixes expert width and comparison
denominator and should not be generalized.

In the new controlled experiment, with the same width and expert = hidden / 2,
A-Net activates about **50%** of the independent baseline's parameters. The
old 95.84% complex-task result is kept in the historical PDF, but uses a
different data/noise setup and is not directly comparable to the new 98.09–
98.24% results.

## Quick start

### Install

~~~bash
pip install -r requirements.txt
~~~

For the tested CUDA 13.0 build on RTX 5060 / Blackwell:

~~~bash
pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install numpy matplotlib pytest
~~~

### Run the scaling experiment

~~~bash
# preflight: about 30 optimizer steps
python -m experiments.scaling.run_scaling \
  --scales small --models independent shared anet cross \
  --seeds 2026 --epochs 2 --max-steps 30 \
  --out-dir experiments/scaling/results/preflight

# full controlled matrix (36 runs)
python -m experiments.scaling.run_scaling \
  --scales small medium large \
  --models independent shared anet cross \
  --seeds 2026 2027 2028 --epochs 50 \
  --out-dir experiments/scaling/results/scaling_v1 --resume

# summaries, CSV and figures
python -m experiments.scaling.analyze_results \
  --results-dir experiments/scaling/results/scaling_v1

# true process-level cold start, one seed
python -m experiments.scaling.benchmark_cold_starts \
  --results-dir experiments/scaling/results/scaling_v1 --seed 2026

# same-parameter cross-layer disabled ablation (9 runs)
python -m experiments.scaling.run_scaling \
  --scales small medium large --models cross_off \
  --seeds 2026 2027 2028 --epochs 50 \
  --out-dir experiments/scaling/results/scaling_cross_ablation
~~~

### Run the earlier demos

~~~bash
# original 3-task demo: CPU / CUDA
python multitask_anet_demo.py --device auto

# earlier complex-task suite
python complex_tasks_test.py --device auto

# CUDA compatibility / GPU smoke test
python cuda_compat_test.py

# CPU inference latency benchmark
python benchmark_latency.py
~~~

## Tests

~~~bash
python -m pytest -q -p no:cacheprovider tests
~~~

Current result: **14 passed**. Tests cover model shapes, parameter formulas,
FLOP-counter agreement, selected-expert gradients, shared-block identity,
gradient accumulation across paths, disabled-call ablation, task determinism
and clean test labels.

## Repository layout

~~~text
.
├── multitask_anet_demo.py              # current A-Net implementation + simple tasks
├── complex_tasks_test.py               # earlier complex-task suite
├── cuda_compat_test.py                 # CUDA checks / GPU smoke test
├── benchmark_latency.py                # CPU latency benchmark
├── requirements.txt
├── experiments/
│   └── scaling/
│       ├── PROTOCOL.md                 # pre-run design and fairness controls
│       ├── REPORT.md                   # final feasibility report
│       ├── models.py                   # model zoo + cross-layer prototype
│       ├── tasks.py                    # controlled task suite
│       ├── run_scaling.py              # training / metrics / checkpoints
│       ├── analyze_results.py          # aggregation + figures
│       ├── benchmark_cold_start.py     # fresh-process cold start child
│       └── benchmark_cold_starts.py    # cold-start batch runner
└── tests/
    ├── test_cross_layer.py
    ├── test_scaling_models.py
    └── test_scaling_tasks.py
~~~

## CUDA / CPU notes

- Verified on NVIDIA RTX 5060 Laptop, sm_120, CUDA 13.0 wheel,
  cuDNN 9.24.0, PyTorch 2.14.1+cu130.
- FP16/BF16/autocast/cuDNN smoke tests pass.
- Simple-task CPU/GPU accuracy matched exactly; complex-task CPU/GPU averages
  differed by at most 0.05 pp.
- CUDA compatibility does not imply speed. This small-model workload is often
  launch-bound; batch=1 can be slower on GPU. See the report for measured
  latency and true cold-start data.

## Limitations

- Synthetic 2D tasks only; no real-image/language/tabular benchmark yet.
- Three seeds; no formal significance test.
- One GPU, one PyTorch/CUDA stack, shared with desktop applications.
- No learned routing (C); no general cross-layer system (B).
- A-Net is not universally faster and does not have lower storage than the
  shared trunk. Its demonstrated benefit is a parameter/compute trade-off at
  near-independent accuracy, with throughput gains only in favorable
  scale/batch regimes.

## License

See [LICENSE](LICENSE). The prior report [A-Net_Experiment_Report_V3.pdf](A-Net_Experiment_Report_V3.pdf)
is retained for history but predates the scaling study.
