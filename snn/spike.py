"""脉冲发放与阈值处的替代梯度。"""

from typing import Protocol

import numpy as np


class SpikeFunction(Protocol):
    """可替换的脉冲函数。两个方法的输入均为 z = u - θ。"""

    def forward(self, z: np.ndarray) -> np.ndarray: ...

    def derivative(self, z: np.ndarray) -> np.ndarray: ...


class FastSigmoidSpike:
    """前向用阶跃函数，反向用 fast sigmoid 的导数。"""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return (z >= 0).astype(z.dtype)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # 阶跃函数在阈值外梯度为零。这个替代导数让阈值附近也能传回梯度。
        return 1.0 / (1.0 + 5.0 * np.abs(z)) ** 2
