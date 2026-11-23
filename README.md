# 用 NumPy 学习 LIF 和 BPTT

这个示例在 CPU 上训练 MNIST 手写数字分类器。网络结构为 `784 维像素 → 128 个隐藏 LIF 神经元 → 10 个输出 LIF 神经元`。默认把每张图像输入 16 个时间步。两个 LIF 层的膜电位逐步变化，每层的权重和偏置在所有时间步共享。

## 运行

需要 Python 3.10 或更高版本，以及 NumPy。

```powershell
python -m pip install -r requirements.txt
python train.py
```

默认训练 3 轮，每批 128 张图像，Adam 学习率为 `0.001`，随机种子为 `42`。每轮打印训练损失、训练准确率和测试准确率。训练准确率是在这一轮逐批训练时统计的，测试准确率使用该轮结束后的参数。

首次运行会下载 `data/mnist.npz`，验证 SHA-256 后保存为本地缓存。也可以通过 `--data-path` 指定原版 `mnist.npz` 的位置。程序沿用原始文件的 60,000 张训练图像和 10,000 张测试图像。数据来源与校验值见 [Keras 的 MNIST 加载器](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py)。

```powershell
python train.py --epochs 1 --train-limit 1024 --test-limit 256
python train.py --data-path 'D:\datasets\mnist.npz'
python train.py --help
```

## 一次训练更新怎么发生

1. **输入编码**。[`snn/encoding.py`](snn/encoding.py) 将图像展平成 `[批量, 784]`，把像素从 `0–255` 缩放到 `[0, 1]`。同一张图像在所有时间步重复输入。隐藏层电流 `I = X @ W1 + b1` 可以一次算好，每步继续更新膜电位。
2. **LIF 前向**。[`snn/lif.py`](snn/lif.py) 先将上一步复位后的电位乘以泄漏系数，再加本步电流。达到阈值就发一个脉冲，并从电位中减去阈值。隐藏层的脉冲通过 `W2` 和 `b2` 形成同一步的输出层电流。两个层都在每个批次从零膜电位开始。
3. **输出读出和损失**。[`snn/model.py`](snn/model.py) 将每个输出神经元在各步的脉冲相加，得到 10 个类别分数。脉冲最多的类别是预测结果，平分时取编号最小的类别。[`snn/loss.py`](snn/loss.py) 计算一次交叉熵和它对这 10 个分数的梯度。
4. **手写 BPTT 与参数更新**。网络先反传输出层，再反传隐藏层。每个 LIF 层按时间逆序算电流的梯度。网络累加各步对共享权重和偏置的梯度，最后由 [`snn/optim.py`](snn/optim.py) 中的 Adam 更新参数。[`train.py`](train.py) 按这个顺序串起每个批次。

## 对照公式读 BPTT

前向使用复位前电位 `u`、脉冲 `s` 和复位后电位 `v`。默认 `β = 0.9`、`θ = 1.0`。

```text
u_t = β v_(t-1) + I_t
s_t = H(u_t - θ)
v_t = u_t - θ s_t
```

反向时，`q_t` 是当前脉冲从本层外部收到的梯度，`δ_t` 是总损失对当前复位前电位 `u_t` 的梯度。用 [`snn/spike.py`](snn/spike.py) 的 `φ_t = 1 / (1 + 5 |u_t - θ|)²` 近似阶跃函数的导数。

```text
δ_(T+1) = 0
δ_t ≈ q_t φ_t + β (1 - θ φ_t) δ_(t+1)
       └ 当前脉冲 ┘  └ 未来经复位和泄漏传回 ┘
```

`1 - θ φ_t` 来自对 `v_t = u_t - θ s_t` 求导。输出层的分数是脉冲和，所以每步的 `q_t` 都等于损失对这个分数的梯度。隐藏层的 `q_t` 则由输出层同一时间步的电流梯度通过 `W2` 传回。由于 `u_t` 对 `I_t` 的局部导数为 1，各步的 `δ_t` 也就是电流梯度。

代码把一个批次的样本放在矩阵的各行，因此用 `X_t.T @ Δ_t` 算每步的权重梯度。对时间步求和，得到共享权重的梯度。偏置梯度是把 `Δ_t` 沿时间和批量两维求和。

## 替换和验证

- 换输入编码时，编码器可以继续返回 `[批量, 特征]`，也可以返回 `[时间步, 批量, 特征]`，给每个时间步不同的输入。
- 换脉冲函数时，实现 `forward(z)` 和 `derivative(z)`，其中 `z = u - θ`，然后传给 `LIF(spike_function=...)`。`SNN` 接受分别配置好的 `hidden_lif` 和 `output_lif`。
- 运行下列命令。测试包括手算的膜电位、脉冲和跨时间梯度，以及把硬阈值换成可微函数后对两层权重和偏置做的有限差分校验。

```powershell
python -m unittest discover -s tests -v
```
