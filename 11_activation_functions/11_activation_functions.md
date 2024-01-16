# 11_activation_functions

## `step function`

Step 函数阶跃函数)是一种常见的激活函数，通常用于人工神经网络中。它是一个分段函数，其输出在输入超过某个阈值时突然跃升到一个常数值，否则保持为另一个常数值。

数学上，Step 函数可以表示为：

$$ \text{step}(x) = \begin{cases} 1, & \text{if } x \geq 0 \\ 0, & \text{if } x < 0 \end{cases} $$

其中，Step 函数的输出在 $x \geq 0$ 时为1，而在 $x < 0$ 时为0。它是一个分段函数，具有两个不同的输出值。

Step 函数通常用于二元分类问题中，其中 1 表示正类别，0 表示负类别。然而，由于它在 $x = 0$ 处不连续，导数在该点处为0(在数学上是不可导的)，因此在深度学习中不常用作隐藏层的激活函数。相反，通常使用连续可导的激活函数，如 Sigmoid、ReLU(Rectified Linear Unit)等。

```python
def step_function(x):
    if x >= 0:
        return 1
    else:
        return 0
```

## `sigmoid`

Sigmoid 函数(又称 Logistic 函数)是一种常用的激活函数，通常用于深度学习中的二元分类问题和某些回归问题。Sigmoid 函数将任意实数映射到一个区间在 0 到 1 之间的数值，它的数学表示如下：

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

其中，$\sigma(x)$ 是 Sigmoid 函数的输出，$e$ 是自然对数的底数，$x$ 是输入。

Sigmoid 函数的性质如下：

- 当 $x$ 趋向正无穷大时，$\sigma(x)$ 趋向 1。
- 当 $x$ 趋向负无穷大时，$\sigma(x)$ 趋向 0。
- 在 $x=0$ 处，$\sigma(0)$ 正好等于 0.5。

Sigmoid 函数具有连续可导的性质，这使得它在梯度下降等优化算法中具有良好的可导性质，可以用于训练神经网络的隐藏层或输出层。

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

## `tanh`

双曲正切函数(Tanh，又称为双曲正切激活函数)是一种常用的激活函数，通常用于神经网络中。Tanh 函数将任何实数映射到区间在 -1 到 1 之间的数值。Tanh 函数的数学表示如下：

$$  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

其中，$\text{tanh}(x)$ 是 Tanh 函数的输出，$e$ 是自然对数的底数，$x$ 是输入。

Tanh 函数的性质如下：

- 当 $x$ 趋向正无穷大时，$\text{tanh}(x)$ 趋向 1。
- 当 $x$ 趋向负无穷大时，$\text{tanh}(x)$ 趋向 -1。
- 在 $x=0$ 处，$\text{tanh}(0)$ 正好等于 0。

Tanh 函数类似于 Sigmoid 函数，但具有更广的输出范围(-1 到 1)，并且是零中心化的(均值为0)，这有助于缓解梯度消失问题。

```python
import numpy as np

def tanh(x):
    return np.tanh(x)
```

## `relu`

ReLU(Rectified Linear Unit)是一种常用的激活函数，广泛应用于深度神经网络中。ReLU 激活函数将所有负数输入映射为零，而将正数输入保持不变。ReLU 函数的数学表示如下：

$$ \text{ReLU}(x) = \max(0, x) $$

其中，$\text{ReLU}(x)$ 是 ReLU 函数的输出，$x$ 是输入。

ReLU 函数的性质如下：

- 当 $x$ 大于等于零时，$\text{ReLU}(x)$ 等于 $x$。
- 当 $x$ 小于零时，$\text{ReLU}(x)$ 等于零。

ReLU 函数的一个主要优点是其简单性，计算高效，而且在许多深度学习任务中表现良好。它有助于缓解梯度消失问题，并且引入了非线性性质，使得神经网络可以学习更复杂的函数。

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)
```

## `leaky relu`

Leaky ReLU(Leaky Rectified Linear Unit)是一种修正线性单元激活函数的变体。与标准的 ReLU 不同，Leaky ReLU 允许小于零的输入值具有一个小的斜率(通常为小的正数)，而不是将它们直接置为零。这有助于缓解标准 ReLU 可能存在的问题，如"死亡 ReLU"问题，其中某些神经元在训练过程中变得不活跃，导致梯度消失。

Leaky ReLU 函数的数学表示如下：

$$ \text{LeakyReLU}(x) = \begin{cases} x, & \text{if } x \geq 0 \\ \alpha x, & \text{if } x < 0 \end{cases} $$

其中，$\text{LeakyReLU}(x)$ 是 Leaky ReLU 函数的输出，$x$ 是输入，$\alpha$ 是一个小正数，通常接近零。

Leaky ReLU 函数的性质如下：

- 当 $x$ 大于等于零时，$\text{LeakyReLU}(x)$ 等于 $x$。
- 当 $x$ 小于零时，$\text{LeakyReLU}(x)$ 等于 $\alpha x$，其中 $\alpha$ 是小正数。

Leaky ReLU 的引入使得神经网络在训练期间能够传递小梯度，从而减轻了一些训练问题。通常，$\alpha$ 的选择是一个超参数，可以根据具体任务进行调整。

```python
import numpy as np

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)
```

## `softmax`

Softmax 函数是一种常用的激活函数，通常用于多类别分类问题中，它将原始分数(也称为 logits)转换成归一化的概率分布。Softmax 函数的数学表示如下：

对于输入向量 $x$，Softmax 函数的输出 $y_i$ 计算如下：

$$ y_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} $$

其中：

- $N$ 是输入向量 $x$ 的维度。
- $x_i$ 是输入向量的第 $i$ 个元素。
- $e$ 是自然对数的底数。

Softmax 函数的作用是将原始分数转换成概率分布，使得每个元素 $x_i$ 被映射为概率 $y_i$，使得所有 $y_i$ 的和等于1。因此，Softmax 函数的输出可以解释为每个类别的概率。

在深度学习中，通常在多类别分类问题的输出层中使用 Softmax 函数，以便将模型的原始输出转换为概率分布。然后，可以根据最高概率的类别作为预测结果。

```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 为了数值稳定性，减去最大值
    return e_x / e_x.sum()
```

## `import torch.nn.functional as F`

`torch.nn.functional` 模块是 PyTorch 中用于深度学习操作的功能性函数模块，它包含了许多常用的函数，用于构建神经网络、定义损失函数、进行激活函数操作、计算梯度等。以下是一些常用的 `torch.nn.functional` 中的函数：

1. **激活函数**：
   - `torch.nn.functional.relu(input, inplace=False)`：ReLU 激活函数。
   - `torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)`：Leaky ReLU 激活函数。
   - `torch.nn.functional.sigmoid(input)`：Sigmoid 激活函数。
   - `torch.nn.functional.tanh(input)`：双曲正切(Tanh)激活函数。
   - `torch.nn.functional.softmax(input, dim=None)`：Softmax 激活函数，用于多类别分类。
2. **损失函数**：
   - `torch.nn.functional.cross_entropy(input, target)`：交叉熵损失函数，通常用于分类问题。
   - `torch.nn.functional.mse_loss(input, target)`：均方误差(MSE)损失函数，通常用于回归问题。
   - `torch.nn.functional.binary_cross_entropy(input, target)`：二元交叉熵损失函数，通常用于二元分类问题。
   - `torch.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False)`：三元损失函数，通常用于度量学习任务。
3. **池化操作**：
   - `torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0)`：最大池化。
   - `torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0)`：平均池化。
4. **卷积操作**：
   - `torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0)`：二维卷积操作。
   - `torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0)`：二维转置卷积操作。
5. **批归一化操作**：
   - `torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)`：批归一化操作。
6. **其他常用函数**：
   - `torch.nn.functional.linear(input, weight, bias=None)`：线性变换。
   - `torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)`：Dropout 操作，用于正则化。
   - `torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)`：嵌入层操作，通常用于将类别编码映射为密集向量。