"""用手算结果与数值梯度核对 LIF 和两层网络的 BPTT。"""

import unittest

import numpy as np

from snn.lif import LIF
from snn.loss import cross_entropy
from snn.model import SNN


class SmoothSpike:
    """有限差分要求前向可微，测试时用连续函数替换硬阈值。"""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return 0.5 + z / (1.0 + 5.0 * np.abs(z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + 5.0 * np.abs(z)) ** 2


class BPTTTests(unittest.TestCase):
    def test_lif_by_hand(self) -> None:
        lif = LIF(beta=0.5, threshold=1.0)
        currents = np.array([0.5, 1.0, 0.625, 0.625]).reshape(4, 1, 1)
        spikes, offsets = lif.forward(currents)
        # 复位前电位 0.5、1.25、0.75、1.0。第二步的 0.25 余量参与第三步计算。
        np.testing.assert_array_equal(offsets[:, 0, 0] + 1.0, [0.5, 1.25, 0.75, 1.0])
        np.testing.assert_array_equal(spikes[:, 0, 0], [0, 1, 0, 1])

        # 只有最后一步有直接贡献，前面的梯度全部来自未来。
        direct = np.array([0.0, 0.0, 0.0, 2.0]).reshape(4, 1, 1)
        d_currents = lif.backward(direct, offsets)
        delta_4 = 2.0
        # 第二、三步 |u - θ| = 0.25，替代导数为 16/81。
        delta_3 = 0.5 * (65 / 81) * delta_4
        delta_2 = 0.5 * (65 / 81) * delta_3
        # 第一步 |u - θ| = 0.5，替代导数为 4/49。
        delta_1 = 0.5 * (45 / 49) * delta_2
        np.testing.assert_allclose(d_currents[:, 0, 0], [delta_1, delta_2, delta_3, delta_4])

    def test_two_layer_finite_difference(self) -> None:
        model = SNN(
            np.random.default_rng(42), input_size=3, hidden_size=2, output_size=2,
            hidden_lif=LIF(spike_function=SmoothSpike()),
            output_lif=LIF(spike_function=SmoothSpike()), dtype=np.float64,
        )
        inputs = np.array([[0.1, 0.4, 0.8], [0.9, 0.2, 0.3]])
        sequence = np.stack([inputs, 0.6 * inputs, inputs[::-1]])
        labels = np.array([0, 1])
        steps = 3
        epsilon = 1e-5

        for batch in (inputs, sequence):
            with self.subTest(input_shape=batch.shape):
                scores, cache = model.forward(batch, steps)
                _, d_scores = cross_entropy(scores, labels)
                gradients = model.backward(d_scores, cache)

                # 对每个参数做中心差分，与反向递推得到的解析梯度比较。
                for name, parameter in model.params.items():
                    numeric = np.empty_like(parameter)
                    for index in np.ndindex(parameter.shape):
                        value = parameter[index]
                        parameter[index] = value + epsilon
                        plus, _ = cross_entropy(model.forward(batch, steps)[0], labels)
                        parameter[index] = value - epsilon
                        minus, _ = cross_entropy(model.forward(batch, steps)[0], labels)
                        parameter[index] = value
                        numeric[index] = (plus - minus) / (2 * epsilon)
                    np.testing.assert_allclose(
                        gradients[name], numeric, atol=1e-8, rtol=1e-5, err_msg=name,
                    )

    def test_constant_inputs_match_repeated_sequence(self) -> None:
        model = SNN(np.random.default_rng(7), input_size=3, hidden_size=4, output_size=2)
        inputs = np.array([[0.2, 0.1, 0.8], [0.6, 0.3, 0.5]], dtype=np.float32)
        steps = 8
        repeated = np.repeat(inputs[None, :, :], steps, axis=0)
        scores, cache = model.forward(inputs, steps)
        repeated_scores, repeated_cache = model.forward(repeated, steps)
        np.testing.assert_array_equal(scores, repeated_scores)

        _, d_scores = cross_entropy(scores, np.array([0, 1]))
        gradients = model.backward(d_scores, cache)
        repeated_gradients = model.backward(d_scores, repeated_cache)
        for name in gradients:
            np.testing.assert_allclose(gradients[name], repeated_gradients[name], atol=1e-7)
