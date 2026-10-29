"""
复杂任务泛化实验：3 个更难的非线性任务 × 3 个随机种子。

任务：
1. 高频正弦：x2 > 0.55*sin(3*x1) + 0.28*sin(7.5*x1)，2% 标签噪声
2. 四同心环：r 落在 4 个环带内，标签按 0/1/0/1 交替，2% 标签噪声
3. 旋转棋盘：25° 旋转后的 1.1 宽网格奇偶标签，2% 标签噪声

训练配置：
- 每任务 train=5,000 / test=1,500
- hidden=128；A-Net expert hidden=64
- Adam lr=1e-3，80 epochs，batch=256
- CPU 8 线程（准确率实验，不做延迟结论）

运行：
    python complex_tasks_test.py
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from multitask_anet_demo import (
    ANetHardRouting,
    IndependentBaseline,
    SharedTrunkWithHeads,
    count_trainable_parameters,
    evaluate_per_task,
    print_table,
    set_seed,
    train_model,
)

SEED_LIST = (2026, 2027, 2028)
N_TRAIN = 5000
N_TEST = 1500
EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3
HIDDEN_DIM = 128
EXPERT_HIDDEN_DIM = 64
NOISE_RATE = 0.02
NUM_TASKS = 3

torch.set_num_threads(8)


def flip_labels(
    y: np.ndarray, rng: np.random.Generator, rate: float = NOISE_RATE
) -> np.ndarray:
    mask = rng.random(y.shape[0]) < rate
    out = y.copy()
    out[mask] = 1 - out[mask]
    return out


def make_high_freq_sine(
    n: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    x = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
    threshold = 0.55 * np.sin(3.0 * x[:, 0]) + 0.28 * np.sin(7.5 * x[:, 0])
    y = (x[:, 1] > threshold).astype(np.int64)
    y = flip_labels(y, rng)
    return torch.from_numpy(x), torch.from_numpy(y)


def make_four_rings(
    n: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    bands = [(0.05, 0.55), (0.75, 1.25), (1.45, 1.95), (2.15, 2.55)]
    labels = [0, 1, 0, 1]
    per_band = n // len(bands)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for (r_lo, r_hi), label in zip(bands, labels):
        r = rng.uniform(r_lo, r_hi, size=per_band)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=per_band)
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))
        _ = label

    x = np.stack([np.concatenate(xs), np.concatenate(ys)], axis=1).astype(np.float32)
    y = np.repeat(np.asarray(labels, dtype=np.int64), per_band)
    y = flip_labels(y, rng)
    perm = rng.permutation(x.shape[0])
    return torch.from_numpy(x[perm]), torch.from_numpy(y[perm])


def make_rotated_checkerboard(
    n: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    x = rng.uniform(-2.0, 2.0, size=(n, 2)).astype(np.float32)
    theta = math.radians(25.0)
    u = x[:, 0] * math.cos(theta) + x[:, 1] * math.sin(theta)
    v = -x[:, 0] * math.sin(theta) + x[:, 1] * math.cos(theta)
    cell = 1.1
    y = (
        (np.floor((u + 4.0) / cell) + np.floor((v + 4.0) / cell)).astype(np.int64) % 2
    )
    y = flip_labels(y, rng)
    return torch.from_numpy(x), torch.from_numpy(y)


TASK_BUILDERS = [make_high_freq_sine, make_four_rings, make_rotated_checkerboard]
TASK_NAMES = ["高频正弦", "四同心环", "旋转棋盘"]


def build_datasets(seed: int):
    train_rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed + 10_000)
    train = [builder(N_TRAIN, train_rng) for builder in TASK_BUILDERS]
    test = [builder(N_TEST, test_rng) for builder in TASK_BUILDERS]
    return train, test


MODEL_SPECS = [
    ("独立基线", IndependentBaseline, {"hidden_dim": HIDDEN_DIM}),
    ("共享主干", SharedTrunkWithHeads, {"hidden_dim": HIDDEN_DIM}),
    (
        "A-Net",
        ANetHardRouting,
        {"hidden_dim": HIDDEN_DIM, "expert_hidden_dim": EXPERT_HIDDEN_DIM},
    ),
]


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("请求了 --device cuda，但 torch.cuda.is_available() == False。")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复杂任务泛化实验")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备：auto 自动选择，默认 auto",
    )
    return parser.parse_args()


def main(device: torch.device) -> None:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        device_label = (
            f"CUDA · {props.name} · cc{props.major}.{props.minor} · "
            f"{props.total_memory / 1024**3:.1f} GiB"
        )
    else:
        device_label = f"CPU（{torch.get_num_threads()} 线程）"

    print("=" * 96)
    print("复杂任务泛化实验：高频正弦 / 四同心环 / 旋转棋盘".center(96))
    print("=" * 96)
    print(f"任务数              : {NUM_TASKS}")
    print(f"每任务样本          : train={N_TRAIN:,} / test={N_TEST:,}")
    print(f"标签噪声            : {NOISE_RATE * 100:.0f}%")
    print(f"随机种子            : {SEED_LIST}")
    print(f"训练配置            : Adam(lr={LR}), epochs={EPOCHS}, batch={BATCH_SIZE}, hidden={HIDDEN_DIM}, A-Net expert={EXPERT_HIDDEN_DIM}")
    print(f"运行设备            : {device_label}")
    print()

    raw: dict[str, dict[str, list[float]]] = {}
    train_times: dict[str, list[float]] = {}
    param_counts: dict[str, tuple[int, int]] = {}

    for seed in SEED_LIST:
        train_data, test_data = build_datasets(seed)
        print(
            f"[seed={seed}] 训练集类别占比: "
            + ", ".join(f"T{i + 1}={float((y == 1).float().mean()):.3f}" for i, (_, y) in enumerate(train_data))
        )
        for name, cls, kwargs in MODEL_SPECS:
            set_seed(seed)
            model = cls(**kwargs).to(device)
            total_params = count_trainable_parameters(model)
            active_params = total_params
            if isinstance(model, ANetHardRouting):
                active_params = count_trainable_parameters(
                    model.l1
                ) + count_trainable_parameters(model.experts[0])
            param_counts.setdefault(name, (total_params, active_params))

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            train_model(
                model,
                train_data,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            accs = evaluate_per_task(model, test_data, device=device)
            raw.setdefault(name, {})[str(seed)] = accs
            train_times.setdefault(name, []).append(elapsed)
            print(
                f"  {name:<8s} acc="
                + ", ".join(f"{a * 100:6.2f}%" for a in accs)
                + f"  mean={sum(accs) / len(accs) * 100:6.2f}%  time={elapsed:5.1f}s"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows: list[list[str]] = []
    summary: dict[str, dict[str, float]] = {}
    mean_acc_by_model: dict[str, list[float]] = {}
    for name, _, _ in MODEL_SPECS:
        total_params, active_params = param_counts[name]

        per_task_means = []
        per_task_stds = []
        for task_idx in range(NUM_TASKS):
            values = np.array([raw[name][str(seed)][task_idx] for seed in SEED_LIST]) * 100
            per_task_means.append(float(values.mean()))
            per_task_stds.append(float(values.std(ddof=1)))
        averages = np.array(
            [
                sum(raw[name][str(seed)]) / NUM_TASKS * 100
                for seed in SEED_LIST
            ]
        )
        mean_acc_by_model[name] = averages.tolist()
        summary[name] = {
            "total_params": total_params,
            "active_params": active_params,
            "task_mean": per_task_means,
            "task_std": per_task_stds,
            "average_mean": float(averages.mean()),
            "average_std": float(averages.std(ddof=1)),
            "train_time_mean": float(np.mean(train_times[name])),
        }
        rows.append(
            [
                name,
                f"{total_params:,d}",
                f"{active_params:,d}",
                f"{per_task_means[0]:.2f} ± {per_task_stds[0]:.2f}",
                f"{per_task_means[1]:.2f} ± {per_task_stds[1]:.2f}",
                f"{per_task_means[2]:.2f} ± {per_task_stds[2]:.2f}",
                f"{averages.mean():.2f} ± {averages.std(ddof=1):.2f}",
            ]
        )

    print()
    print(f"复杂任务测试集准确率（3 seeds，mean ± std，{device_label}）")
    print_table(
        ["方案", "总参数", "激活参数", "任务1 高频正弦", "任务2 四同心环", "任务3 旋转棋盘", "平均"],
        rows,
    )
    print()
    print(f"数据量合计: {NUM_TASKS * (N_TRAIN + N_TEST):,} 个样本")
    print(f"平均训练耗时: " + ", ".join(f"{name}={summary[name]['train_time_mean']:.1f}s" for name in summary))

    output = {
        "config": {
            "seeds": list(SEED_LIST),
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "hidden_dim": HIDDEN_DIM,
            "expert_hidden_dim": EXPERT_HIDDEN_DIM,
            "noise_rate": NOISE_RATE,
            "task_names": TASK_NAMES,
        },
        "summary": summary,
        "raw": raw,
    }
    output["device"] = device_label
    out_path = f".complex_results_{device.type}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(resolve_device(args.device))
