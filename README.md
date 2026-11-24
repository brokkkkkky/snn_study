# 用 NumPy 学习 LIF 和 BPTT

这个示例在 CPU 上训练 MNIST 手写数字分类器。网络结构为 `784 维像素 → 128 个隐藏 LIF 神经元 → 10 个输出 LIF 神经元`。默认把每张图像输入 16 个时间步。两个 LIF 层的膜电位逐步变化，每层的权重和偏置在所有时间步共享。

- [参考链接 1](https://chatgpt.com/s/cx_6b05242710108191a048636b3acaacd2)
- [参考链接 2](https://chatgpt.com/s/cx_6b052450f7c08191a3461b80a0ba23ab)

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

### 1. LIF 前向与替代导数

$u_t$ 是复位前膜电位，$s_t$ 是脉冲，$v_t$ 是复位后膜电位。初始电位 $v_0=0$，默认泄漏系数 $\beta=0.9$，发放阈值 $\theta=1.0$。

$$
\begin{aligned}
u_t &= \beta v_{t-1} + I_t, \\
s_t &= H(u_t-\theta), \\
v_t &= u_t - \theta s_t.
\end{aligned}
$$

达到阈值时，阶跃函数 $H$ 输出 $1$，随后电位减去一个阈值。反向时，用 $\phi_t$ 近似阶跃函数在该处的导数。 [snn/spike.py](snn/spike.py)

$$
\phi_t = \frac{1}{\left(1+5\lvert u_t-\theta\rvert\right)^2}.
$$

### 2. 交叉熵给出输出层的直接梯度

第 $b$ 张图像的第 $k$ 类分数 $C_{b,k}$ 是该输出神经元在 $T$ 步内的脉冲总数。对每张图像的分数做 softmax 得到概率 $p_{b,k}$，再用真实标签 $y_b$ 计算批量平均交叉熵。

$$
\begin{aligned}
C_{b,k} &= \sum_{t=1}^{T} s^{\mathrm{out}}_{t,b,k}, \\
p_{b,k} &= \frac{\exp(C_{b,k})}{\sum_{j=0}^{9}\exp(C_{b,j})}, \\
L &= -\frac{1}{B}\sum_{b=1}^{B}\log p_{b,y_b}.
\end{aligned}
$$

`d_counts` 是 $\partial L/\partial C$。每个脉冲以系数 $1$ 加入总脉冲数，因此同一个样本、同一个输出神经元的每个时间步都收到相同的直接梯度 $q^{\mathrm{out}}_{t,b,k}$。 [snn/loss.py](snn/loss.py)

$$
q^{\mathrm{out}}_{t,b,k}
= \frac{\partial L}{\partial C_{b,k}}
= \frac{p_{b,k}-\mathbf{1}[k=y_b]}{B}.
$$

### 3. 从最后一个时间步递推

对单个 LIF 神经元，$q_t$ 表示从本层外部传到脉冲 $s_t$ 的梯度，$\delta_t$ 表示损失对复位前膜电位 $u_t$ 的梯度。复位式给出 $1-\theta\phi_t$，下一步的膜电位更新再带来泄漏系数 $\beta$。

$$
\begin{aligned}
\delta_{T+1} &= 0, \\
\delta_t &\approx
\underbrace{q_t\phi_t}_{\mathrm{direct}}
+
\underbrace{\beta(1-\theta\phi_t)\,\delta_{t+1}}_{\mathrm{future}},
\qquad t=T,\ldots,1.
\end{aligned}
$$

末步只剩直接项 $\delta_T\approx q_T\phi_T$。本步的 $I_t$ 以系数 $1$ 加入 $u_t$，所以 $\delta_t$ 也就是电流梯度。 [snn/lif.py](snn/lif.py)

代码按行存放一个批次的样本。用 $\Delta_t$ 表示一批样本在时间步 $t$ 的电流梯度，用 $Q_t$ 表示同一批样本的直接脉冲梯度。输出层先用交叉熵给出的 $Q_t^{\mathrm{out}}$ 完成 BPTT，再把该步的完整 $\Delta_t^{\mathrm{out}}$ 通过 $W_2^{\mathsf T}$ 传给隐藏层。 [snn/model.py](snn/model.py)

$$
Q_t^{\mathrm{hid}} = \Delta_t^{\mathrm{out}} W_2^{\mathsf T}.
$$

这个 $Q_t^{\mathrm{hid}}$ 用于隐藏层递推中的直接项。隐藏层再用自己的替代导数与跨时间项，算出 $\Delta_t^{\mathrm{hid}}$。

### 4. 累加共享权重的梯度

$X_t$ 是第一层在第 $t$ 步的输入，$S_t^{\mathrm{hid}}$ 是隐藏层在同一步的脉冲，也是第二层的输入。每个时间步的矩阵乘法汇总了该批样本的贡献，再对所有时间步求和。

$$
\begin{aligned}
\frac{\partial L}{\partial W_1}
&\approx \sum_{t=1}^{T} X_t^{\mathsf T}\Delta_t^{\mathrm{hid}}, \\
\frac{\partial L}{\partial W_2}
&\approx \sum_{t=1}^{T} \left(S_t^{\mathrm{hid}}\right)^{\mathsf T}
\Delta_t^{\mathrm{out}}.
\end{aligned}
$$

默认 MNIST 编码让所有 $X_t$ 都等于同一份 $X$，因此代码先加总隐藏层的时间梯度，再用 `inputs.T` 左乘。偏置梯度直接在时间和批量两个维度上对 $\Delta_t$ 求和。

## 替换和验证

- 换输入编码时，编码器可以继续返回 `[批量, 特征]`，也可以返回 `[时间步, 批量, 特征]`，给每个时间步不同的输入。
- 换脉冲函数时，实现 `forward(z)` 和 `derivative(z)`，其中 `z = u - θ`，然后传给 `LIF(spike_function=...)`。`SNN` 接受分别配置好的 `hidden_lif` 和 `output_lif`。
- 运行下列命令。测试包括手算的膜电位、脉冲和跨时间梯度，以及把硬阈值换成可微函数后对两层权重和偏置做的有限差分校验。

```powershell
python -m unittest discover -s tests -v
```
