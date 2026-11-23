"""把 MNIST 像素转换成网络的输入。"""

import numpy as np


def encode_pixels(images: np.ndarray) -> np.ndarray:
    """将 [批量, 28, 28] 的 uint8 像素转成 [批量, 784] 的 float32 数组。"""
    pixels = images.reshape(-1, 784).astype(np.float32)
    # 模型收到二维数组时，会在所有时间步复用同一份输入。
    return pixels / 255.0
