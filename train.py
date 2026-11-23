"""在 CPU 上训练 MNIST 分类器，演示完整的手写 BPTT 流程。"""

import argparse
from pathlib import Path

import numpy as np

from snn.data import load_mnist
from snn.encoding import encode_pixels
from snn.loss import cross_entropy
from snn.model import SNN
from snn.optim import Adam


def run_epoch(
    model: SNN,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    """按给定顺序处理一个数据集。传入优化器时训练，否则只计算指标。"""
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        targets = labels[batch]
        inputs = encode_pixels(images[batch])

        # 前向展开 16 步（可调），再对累计脉冲数计算一次交叉熵。
        counts, cache = model.forward(inputs, steps)
        loss, d_counts = cross_entropy(counts, targets)
        if optimizer is not None:
            gradients = model.backward(d_counts, cache)
            optimizer.step(gradients)

        # 最后一批可能较小。按实际样本数加权，才是整个数据集的平均损失。
        loss_sum += loss * len(batch)
        correct += int(np.count_nonzero(counts.argmax(axis=1) == targets))

    return loss_sum / len(indices), correct / len(indices)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, help="已有的 MNIST .npz 文件")
    parser.add_argument("--train-limit", type=positive_int, help="只使用训练集的前 N 个样本")
    parser.add_argument("--test-limit", type=positive_int, help="只使用测试集的前 N 个样本")
    parser.add_argument("--epochs", type=positive_int, default=3, help="训练轮数，默认 3")
    parser.add_argument("--steps", type=positive_int, default=16, help="每张图像的时间步数，默认 16")
    parser.add_argument("--batch-size", type=positive_int, default=128, help="批量大小，默认 128")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam 学习率，默认 0.001")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate 必须是有限的正数。")
    if args.seed < 0:
        parser.error("--seed 必须是非负整数。")

    try:
        (train_images, train_labels), (test_images, test_labels) = load_mnist(args.data_path)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"MNIST 加载失败：{exc}\n")

    train_size = len(train_labels[: args.train_limit])
    test_indices = np.arange(len(test_labels[: args.test_limit]))
    rng = np.random.default_rng(args.seed)
    model = SNN(rng)
    optimizer = Adam(model.params, learning_rate=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        # 每轮重新打乱训练样本。测试始终按原顺序进行，且没有参数更新。
        train_loss, train_accuracy = run_epoch(
            model,
            train_images,
            train_labels,
            rng.permutation(train_size),
            steps=args.steps,
            batch_size=args.batch_size,
            optimizer=optimizer,
        )
        _, test_accuracy = run_epoch(
            model,
            test_images,
            test_labels,
            test_indices,
            steps=args.steps,
            batch_size=args.batch_size,
        )
        print(
            f"第 {epoch}/{args.epochs} 轮 | 训练损失 {train_loss:.4f} | "
            f"训练准确率 {train_accuracy:.2%} | 测试准确率 {test_accuracy:.2%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
