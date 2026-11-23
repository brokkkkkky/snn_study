"""交叉熵及其对类别分数的手写梯度。"""

import numpy as np


def cross_entropy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """接收 [批量, 类别] 的分数，以及 [批量] 的整数标签。"""
    # 每行减去最大值不会改变 softmax，且能避免指数溢出。
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    normalizer = probabilities.sum(axis=1, keepdims=True)
    rows = np.arange(len(labels))
    loss = (np.log(normalizer[:, 0]) - shifted[rows, labels]).mean()

    # 交叉熵对分数的梯度是 softmax 概率减去 one-hot 标签。
    # 损失按批量取均值，因此梯度也要除以批量大小。
    probabilities /= normalizer
    probabilities[rows, labels] -= 1.0
    return float(loss), probabilities / len(labels)
