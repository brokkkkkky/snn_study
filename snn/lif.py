"""LIF 的时间展开与减阈值复位，包含手写的时间反向传播。"""

import numpy as np

from .spike import FastSigmoidSpike, SpikeFunction


class LIF:
    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        spike_function: SpikeFunction | None = None,
    ) -> None:
        self.beta = beta
        self.threshold = threshold
        self.spike_function = FastSigmoidSpike() if spike_function is None else spike_function

    def forward(self, currents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """输入电流、脉冲和缓存的阈值偏差均为 [时间步, 批量, 神经元]。"""
        dtype = np.result_type(currents.dtype, np.float32)

        # 初始化
        voltage = np.zeros(currents.shape[1:], dtype=dtype)
        spikes = np.empty(currents.shape, dtype=dtype)
        threshold_offsets = np.empty_like(spikes)

        # current代表当前电流
        for t, current in enumerate(currents):
            # 上一时刻复位后的电位先泄漏，再叠加当前输入电流。
            # beta 是保留率
            pre_reset_voltage = self.beta * voltage + current

            # 反向只需要 u_t - θ，据此重新算该步的替代导数。
            threshold_offsets[t] = pre_reset_voltage - self.threshold

            # 判断是否发放脉冲
            spikes[t] = self.spike_function.forward(threshold_offsets[t])

            # s_t 为 1 时减去阈值，超出阈值的部分留给下一步。
            voltage = pre_reset_voltage - self.threshold * spikes[t]

        return spikes, threshold_offsets


    def backward(
        self, d_spikes: np.ndarray, threshold_offsets: np.ndarray
    ) -> np.ndarray:
        """将来自本层外部的脉冲梯度 q_t 转成每步电流的梯度 δ_t。"""
        d_currents = np.empty_like(threshold_offsets)

        # δ_(T+1) = 0。最后一步之后没有通过膜电位传回来的梯度。
        d_next_u = np.zeros_like(threshold_offsets[0])

        for t in range(len(threshold_offsets) - 1, -1, -1):

            # 计算替代梯度
            phi = self.spike_function.derivative(threshold_offsets[t])
            # d_spikes[t] 是 aL/aS
            # direct 是当前脉冲带来的梯度
            direct = d_spikes[t] * phi

            # 未来状态带来的梯度
            # v_t = u_t - θ s_t，所以 dv_t / du_t ≈ 1 - θ φ_t。
            # u_(t+1) = β v_t + I_(t+1)，再乘 β 就得到跨时间的梯度。
            future = self.beta * (1.0 - self.threshold * phi) * d_next_u
            
            d_u = direct + future
            # u_t 对 I_t 的局部导数为 1，故 dL/dI_t 等于 δ_t。
            d_currents[t] = d_u
            d_next_u = d_u

        return d_currents
