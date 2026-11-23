"""根据手写反向传播给出的梯度，用 Adam 更新参数。"""

import numpy as np


class Adam:
    def __init__(
        self,
        params: dict[str, np.ndarray],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.params = params
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment = {name: np.zeros_like(value) for name, value in params.items()}
        self.second_moment = {name: np.zeros_like(value) for name, value in params.items()}
        self.iteration = 0

    def step(self, gradients: dict[str, np.ndarray]) -> None:
        self.iteration += 1
        first_correction = 1.0 - self.beta1 ** self.iteration
        second_correction = 1.0 - self.beta2 ** self.iteration

        for name, parameter in self.params.items():
            gradient = gradients[name]
            # 分别记录梯度和梯度平方的指数移动平均。
            first = self.beta1 * self.first_moment[name] + (1.0 - self.beta1) * gradient
            second = self.beta2 * self.second_moment[name] + (1.0 - self.beta2) * gradient**2
            self.first_moment[name] = first
            self.second_moment[name] = second
            # 两个移动平均初始为零，先做偏差修正，再原地更新模型参数。
            parameter -= self.learning_rate * (first / first_correction) / (
                np.sqrt(second / second_correction) + self.epsilon
            )
