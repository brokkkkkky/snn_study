"""两层全连接 LIF 网络，输出分数为各类别神经元的总脉冲数。"""

from dataclasses import dataclass

import numpy as np

from .lif import LIF


@dataclass
class ForwardCache:
    """反向计算参数梯度和替代导数所需的前向数据。"""

    inputs: np.ndarray
    hidden_spikes: np.ndarray
    hidden_offsets: np.ndarray
    output_offsets: np.ndarray


class SNN:
    def __init__(
        self,
        rng: np.random.Generator,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        hidden_lif: LIF | None = None,
        output_lif: LIF | None = None,
        dtype: type[np.floating] = np.float32,
    ) -> None:
        self.hidden_lif = LIF() if hidden_lif is None else hidden_lif
        self.output_lif = LIF() if output_lif is None else output_lif

        # 输入使用行向量，因此权重按 [输入特征数, 输出神经元数] 存放。
        self.params = {
            "W1": rng.normal(0, input_size ** -0.5, (input_size, hidden_size)).astype(dtype),
            "b1": np.zeros(hidden_size, dtype=dtype),
            "W2": rng.normal(0, hidden_size ** -0.5, (hidden_size, output_size)).astype(dtype),
            # 让初始化时输出电位在默认仿真长度内能接近阈值。
            "b2": np.full(output_size, 0.15, dtype=dtype),
        }

    def forward(self, inputs: np.ndarray, steps: int) -> tuple[np.ndarray, ForwardCache]:
        """输入为 [批量, 特征] 或 [时间步, 批量, 特征]，返回 [批量, 类别] 分数。"""
        if inputs.ndim not in (2, 3) or steps < 1:
            raise ValueError("输入应为 [批量, 特征] 或 [时间步, 批量, 特征]，时间步数应为正。")
        if inputs.ndim == 3 and inputs.shape[0] != steps:
            raise ValueError("输入序列长度必须等于 steps。")

        hidden_current = inputs @ self.params["W1"] + self.params["b1"]
        if inputs.ndim == 2:
            # 恒定输入的线性投影只需算一次。广播得到各步电流，膜电位仍逐步更新。
            hidden_current = np.broadcast_to(hidden_current, (steps, *hidden_current.shape))
        hidden_spikes, hidden_offsets = self.hidden_lif.forward(hidden_current)

        # 每个时间步，输出层接收同一时间步的隐藏脉冲。
        steps, batch, hidden_size = hidden_spikes.shape
        flat_hidden = hidden_spikes.reshape(steps * batch, hidden_size)
        output_current = flat_hidden @ self.params["W2"] + self.params["b2"]
        output_spikes, output_offsets = self.output_lif.forward(
            output_current.reshape(steps, batch, -1)
        )
        counts = output_spikes.sum(axis=0)
        return counts, ForwardCache(inputs, hidden_spikes, hidden_offsets, output_offsets)

    def backward(self, d_counts: np.ndarray, cache: ForwardCache) -> dict[str, np.ndarray]:
        """先求输出层的时间梯度，再求隐藏层的时间梯度，最后累加参数梯度。"""
        steps, batch, hidden_size = cache.hidden_spikes.shape
        # 计数 C = Σ_t s_t，所以每步都从 C 收到同一个梯度，不除以步数。
        d_output_spikes = np.broadcast_to(d_counts, cache.output_offsets.shape)
        d_output_current = self.output_lif.backward(d_output_spikes, cache.output_offsets)
        flat_d_output = d_output_current.reshape(steps * batch, -1)

        # 在同一时刻，通过 W2 把输出层的电流梯度传给隐藏层的脉冲。
        d_hidden_spikes = (flat_d_output @ self.params["W2"].T).reshape(
            steps, batch, hidden_size
        )
        d_hidden_current = self.hidden_lif.backward(d_hidden_spikes, cache.hidden_offsets)

        # 行向量布局把截图中的 δ_t x_t^T 写为 X_t^T Δ_t。
        # 合并时间和批量后做一次矩阵乘法，等价于将每一步、每个样本的贡献相加。
        if cache.inputs.ndim == 2:
            # x 在所有时刻相同，因此 Σ_t x^T Δ_t = x^T (Σ_t Δ_t)。
            d_w1 = cache.inputs.T @ d_hidden_current.sum(axis=0)
        else:
            flat_inputs = cache.inputs.reshape(steps * batch, -1)
            d_w1 = flat_inputs.T @ d_hidden_current.reshape(steps * batch, hidden_size)
        flat_hidden = cache.hidden_spikes.reshape(steps * batch, hidden_size)
        return {
            "W1": d_w1,
            "b1": d_hidden_current.sum(axis=(0, 1)),
            "W2": flat_hidden.T @ flat_d_output,
            "b2": flat_d_output.sum(axis=0),
        }
