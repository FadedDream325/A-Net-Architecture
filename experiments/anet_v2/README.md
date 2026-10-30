# A-Net v2

This directory is the new, mechanism-first A-Net implementation. It does not
modify the legacy implementation in the repository root or
experiments/scaling/.

## What is implemented

- core/parameter_block.py: ParameterBlock, BlockSpec, ParameterRegistry.
  A block has one owner, one storage location, a monotonic version, and explicit
  use grants. Other workflows call the same object; no parameter copy is made.
- core/workflow.py: Workflow, CallSite, WorkflowGraph, PreL1, TerminalL1.
  The graph is checked for cycles. Terminal L1 receives only workflow-level
  output tensors.
- core/result_cache.py: ResultCache keyed by block identity, block version,
  input fingerprint, and operation configuration. The default input key is a
  cheap object-identity key; a content-hash mode is available for correctness
  checks and for repeated-content workloads. Training caches are cleared at
  every optimizer step; cached tensors keep their autograd graph.
- core/accounting.py: five separate quantities: reference parameters, unique
  parameters, active parameters, theoretical FLOPs, and actual executions/FLOPs.
- core/models.py: Independent, Shared Backbone, A-Net Basic, A-Net Parameter
  Sharing, A-Net Result Reuse, A-Net Full, and adapter variants.
- core/tasks.py: binary nonlinear classification, nonlinear regression, and
  5-class nonlinear classification, with low/medium/high latent correlation.
- core/train.py: unified training/evaluation for all models.
- core/benchmark.py: CUDA-event/CPU latency, true wall-time throughput,
  GPU-resident input assertions, peak memory, raw per-call samples.
- runners/run_stage1.py: Stage 1 width-scaling runner.
- analysis/aggregate_stage1.py: summary CSV/JSON/Markdown and figures.

## Design choices

- Workflow output width is hidden_dim // 2; each workflow has an owned block
  and a cross block of the same width. With hidden_dim width, A-Net Basic's
  active parameter count is close to the Independent baseline, making the
  comparison less confounded by capacity.
- Sharing modes:
  - none: every workflow owns its cross block;
  - block: all workflows call one cross block owned by w0;
  - adapter: shared cross block plus a per-workflow low-rank adapter, counted
    separately.
- Active-workflow modes:
  - sparse: one workflow per sample (used for Stage 1 scaling);
  - pair: two workflows per sample (used to create repeated shared calls);
  - dense: all workflows per sample.
- cache_key_mode="identity" is the performance default. It is safe for
  same-object calls but will miss equal-content different objects; use
  "content" when correctness requires content-level reuse.

## Tests

All tests use CPU and run from the repository root:

    python -m pytest -q -p no:cacheprovider tests

The v2 tests cover parameter identity, gradient accumulation, parameter
versioning, cache correctness/invalidation/gradient, DAG acyclicity, terminal
integration, accounting, and data splits. Legacy tests remain in the suite.

## Stage 1 command

    python -m experiments.anet_v2.runners.run_stage1 \
      --widths 64 128 256 512 1024 2048 \
      --models independent shared anet_basic anet_full \
      --correlations low medium high \
      --seeds 2026 2027 2028 2029 2030 \
      --epochs 30 \
      --out-dir experiments/anet_v2/results/stage1 --resume

Then:

    python -m experiments.anet_v2.analysis.aggregate_stage1 \
      --results-dir experiments/anet_v2/results/stage1

Stage 1 uses sparse active mode. It measures the base workflow/scaling
question; it does not exercise cross-workflow cache hits (there is only one
active workflow). Stage 2 will run pair/dense and the share/adapter ablation;
Stage 3 will drive repeated inputs through the cache.

## Provenance

Every run records the git commit, tracked-diff hash, git status, a SHA256 of
the v2 Python sources and tests, the full config, data hashes, environment,
and raw latency samples. Results are append-only JSONL; summaries are derived
files. The legacy scaling results remain untouched.


## Router and cache refactor (2026-10-30)

- Routing is resolved once at construction: _active_table and direct block
  references (_prel1_block, _own_blocks, _cross_blocks, _adapter_blocks).
  The hot path uses tuple/list indexing, not per-forward if/else task
  prediction.
- TerminalL1 uses nn.ModuleList heads and a K=1 fast path.
- forward_mixed uses torch.argsort(task_ids, stable=True), computes contiguous
  task groups, then applies the inverse permutation. There is no per-sample
  Python mask and no warp-level task branching.
- ANetModel.set_inference_cache(False) is the default for eval: a single
  request runs the pure matrix-multiply path with no cache key construction or
  lookup. set_inference_cache(True) enables request-level cache reuse.
- benchmark_inference(..., inference_cache=False) and the runner flag
  --inference-cache/--no-inference-cache expose the switch.
- core/device_guard.py refuses to start below --min-free-vram-gb (default
  2.0 GiB) and caps the PyTorch allocator at --max-vram-fraction (default
  0.80), so experiments do not silently spill into Windows shared GPU memory.
- Each run records vram_before, vram_after, gpu_peak_allocated_bytes,
  host_to_device_bytes, and input_device.

Refactor benchmark command:

    python -m experiments.anet_v2.runners.benchmark_router_cache \
      --widths 64 256 1024 2048 --seeds 2026 2027 2028 \
      --out-dir experiments/anet_v2/results/refactor_benchmark
