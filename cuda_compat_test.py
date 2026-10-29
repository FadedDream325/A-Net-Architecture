"""
RTX 5060 Laptop / CUDA 兼容性与性能冒烟测试。

覆盖：
1. CUDA 运行时、驱动、cuDNN、BF16、目标架构
2. FP32 / FP16 / BF16 矩阵乘法
3. 三种模型结构在 GPU 上的 forward / backward / Adam step
4. A-Net 硬路由在 GPU 上只更新被选中专家
5. 5 个 epoch 的 GPU 端到端小训练
6. batch=1/64/256 的 GPU 推理延迟与吞吐

运行：
    python cuda_compat_test.py
"""

from __future__ import annotations

import gc
import platform
import statistics
import time
import traceback
from collections.abc import Callable

import torch
import torch.nn.functional as F

from multitask_anet_demo import (
    ANetHardRouting,
    IndependentBaseline,
    SharedTrunkWithHeads,
    build_datasets,
)

DEVICE = torch.device("cuda")
FAILURES: list[str] = []


def fmt_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def check(name: str, fn: Callable[[], str]) -> None:
    try:
        detail = fn()
        print(f"[PASS] {name}" + (f" | {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(name)
        print(f"[FAIL] {name} | {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)


def test_basic_allocation() -> str:
    x = torch.arange(16, device=DEVICE, dtype=torch.float32)
    y = (x * 2 + 1).sum().item()
    torch.cuda.synchronize()
    assert y == 256.0, y
    assert x.device.type == "cuda"
    return f"sum={y:.1f}, device={x.device}"


def test_fp32_matmul() -> str:
    torch.manual_seed(7)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = a @ b
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = (a.to(DEVICE) @ b.to(DEVICE)).cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    max_abs = (out - ref).abs().max().item()
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 1e-4, rel
    return f"max_abs={max_abs:.2e}, rel={rel:.2e}"


def test_fp16_matmul() -> str:
    torch.manual_seed(8)
    a = torch.randn(512, 512, device=DEVICE, dtype=torch.float16)
    b = torch.randn(512, 512, device=DEVICE, dtype=torch.float16)
    out = a @ b
    torch.cuda.synchronize()
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    return f"dtype={out.dtype}, finite=True"


def test_bf16_matmul() -> str:
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("torch.cuda.is_bf16_supported() == False")
    torch.manual_seed(9)
    a = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(512, 512, device=DEVICE, dtype=torch.bfloat16)
    out = a @ b
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    return f"dtype={out.dtype}, finite=True"


def test_cudnn_conv() -> str:
    torch.manual_seed(10)
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(DEVICE)
    x = torch.randn(8, 3, 32, 32, device=DEVICE)
    y = conv(x)
    loss = y.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert y.shape == (8, 8, 32, 32), y.shape
    assert torch.isfinite(loss)
    assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
    return f"out={tuple(y.shape)}, loss={loss.item():.4f}"


def test_model_steps() -> str:
    specs = [
        ("IndependentBaseline", IndependentBaseline),
        ("SharedTrunkWithHeads", SharedTrunkWithHeads),
        ("ANetHardRouting", ANetHardRouting),
    ]
    details: list[str] = []
    for name, cls in specs:
        torch.manual_seed(11)
        model = cls().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(256, 2, device=DEVICE)
        y = torch.randint(0, 2, (256,), device=DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, 0)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        assert logits.shape == (256, 2), logits.shape
        assert torch.isfinite(loss), loss
        assert all(p.device.type == "cuda" for p in model.parameters())

        if isinstance(model, ANetHardRouting):
            assert model.experts[0][0].weight.grad is not None
            assert model.experts[1][0].weight.grad is None
            assert model.experts[2][0].weight.grad is None
            details.append(f"{name}: loss={loss.item():.4f}, only expert0 grad")
        else:
            details.append(f"{name}: loss={loss.item():.4f}")

        del model, optimizer, x, y, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    return "; ".join(details)


def test_autocast_fp16() -> str:
    torch.manual_seed(12)
    model = SharedTrunkWithHeads().to(DEVICE)
    x = torch.randn(256, 2, device=DEVICE)
    y = torch.randint(0, 2, (256,), device=DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x, 0)
        loss = F.cross_entropy(logits, y)
    torch.cuda.synchronize()
    assert torch.isfinite(loss), loss
    return f"autocast loss={loss.item():.4f}"


def quick_train() -> None:
    train_data, test_data = build_datasets()
    train_gpu = [(x.to(DEVICE), y.to(DEVICE)) for x, y in train_data]
    test_gpu = [(x.to(DEVICE), y.to(DEVICE)) for x, y in test_data]

    specs = [
        ("IndependentBaseline", IndependentBaseline),
        ("SharedTrunkWithHeads", SharedTrunkWithHeads),
        ("ANetHardRouting", ANetHardRouting),
    ]
    print("\nGPU 端到端小训练（5 epochs，仅验证兼容性）")
    for name, cls in specs:
        torch.manual_seed(13)
        model = cls().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(5):
            for task_id, (x, y) in enumerate(train_gpu):
                permutation = torch.randperm(x.size(0), device=DEVICE)
                for offset in range(0, x.size(0), 128):
                    idx = permutation[offset : offset + 128]
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x[idx], task_id), y[idx])
                    loss.backward()
                    optimizer.step()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        model.eval()
        accs: list[float] = []
        with torch.inference_mode():
            for task_id, (x, y) in enumerate(test_gpu):
                pred = model(x, task_id).argmax(dim=1)
                accs.append((pred == y).float().mean().item())

        mean_acc = sum(accs) / len(accs)
        print(
            f"  {name:<24s} | 5 epoch 用时 {elapsed:6.2f}s | "
            f"测试准确率 {accs[0]*100:6.2f}% / {accs[1]*100:6.2f}% / "
            f"{accs[2]*100:6.2f}% | 平均 {mean_acc*100:6.2f}%"
        )

        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def benchmark_gpu() -> None:
    specs = [
        ("IndependentBaseline", IndependentBaseline),
        ("SharedTrunkWithHeads", SharedTrunkWithHeads),
        ("ANetHardRouting", ANetHardRouting),
    ]
    batches = (1, 64, 256, 1024, 4096)
    iterations = {1: 1000, 64: 1000, 256: 500, 1024: 300, 4096: 200}
    warmup = 200

    print("\nGPU 推理延迟（CUDA Event，P50；整个 batch 的耗时）")
    header = f"{'模型':<24s} {'batch':>6s} {'P50(μs)':>10s} {'P95(μs)':>10s} {'吞吐(samples/s)':>18s}"
    print(header)
    print("-" * len(header))

    for name, cls in specs:
        torch.manual_seed(14)
        model = cls().to(DEVICE).eval()
        for batch_size in batches:
            x = torch.randn(batch_size, 2, device=DEVICE)
            with torch.inference_mode():
                for _ in range(warmup):
                    model(x, 0)
                torch.cuda.synchronize()

                starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations[batch_size])]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations[batch_size])]
                wall_start = time.perf_counter()
                for i in range(iterations[batch_size]):
                    starts[i].record()
                    model(x, 0)
                    ends[i].record()
                torch.cuda.synchronize()
                wall_elapsed = time.perf_counter() - wall_start

            times_us = [s.elapsed_time(e) * 1000 for s, e in zip(starts, ends)]
            times_us.sort()
            p50 = statistics.median(times_us)
            p95 = times_us[min(len(times_us) - 1, int(len(times_us) * 0.95))]
            throughput = batch_size * iterations[batch_size] / wall_elapsed
            print(
                f"{name:<24s} {batch_size:>6d} {p50:>10.1f} {p95:>10.1f} {throughput:>18,.0f}"
            )

        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    print("=" * 92)
    print("RTX 5060 Laptop / CUDA 兼容性测试".center(92))
    print("=" * 92)
    print(f"Python              : {platform.python_version()}")
    print(f"PyTorch             : {torch.__version__}")
    print(f"编译 CUDA 版本      : {torch.version.cuda}")
    print(f"cuDNN               : {torch.backends.cudnn.version()}")
    print(f"CUDA 可用           : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[FATAL] torch.cuda.is_available() == False，无法继续 GPU 测试。")
        print("请检查 NVIDIA 驱动与 CUDA 版 PyTorch 是否匹配。")
        return 2

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    arch_list = torch.cuda.get_arch_list()
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"GPU                 : {props.name}")
    print(f"Compute Capability  : {props.major}.{props.minor} (sm_{props.major}{props.minor})")
    print(f"SM 数量             : {props.multi_processor_count}")
    print(f"显存                : {fmt_bytes(total_mem)} total / {fmt_bytes(free_mem)} free")
    print(f"PyTorch 编译架构    : {', '.join(arch_list)}")
    print(f"支持 sm_120         : {'yes' if any('120' in a for a in arch_list) else 'no (可能走 PTX)'}")
    print(f"BF16 支持           : {torch.cuda.is_bf16_supported()}")
    print(f"TF32 默认           : {torch.backends.cuda.matmul.allow_tf32}")
    print()

    check("基础显存分配与数据搬运", test_basic_allocation)
    check("FP32 矩阵乘法精度", test_fp32_matmul)
    check("FP16 矩阵乘法", test_fp16_matmul)
    check("BF16 矩阵乘法", test_bf16_matmul)
    check("cuDNN 卷积 forward/backward", test_cudnn_conv)
    check("三种模型 forward/backward/Adam", test_model_steps)
    check("FP16 autocast", test_autocast_fp16)

    quick_train()
    benchmark_gpu()

    print("\n" + "=" * 92)
    print("结果汇总")
    print("=" * 92)
    print(f"失败项数量: {len(FAILURES)}")
    if FAILURES:
        print("失败项: " + ", ".join(FAILURES))
        print("CUDA 兼容性: FAIL")
        return 1
    print("CUDA 兼容性: PASS")
    print(
        f"峰值显存（本进程）: allocated={fmt_bytes(torch.cuda.max_memory_allocated())}, "
        f"reserved={fmt_bytes(torch.cuda.max_memory_reserved())}"
    )
    print("说明：GPU 同时承载桌面应用，延迟绝对值会受系统负载影响；P50 比单次测量更稳定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
