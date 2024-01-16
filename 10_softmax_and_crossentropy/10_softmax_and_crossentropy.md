# 10_softmax_and_crossentropy

## `softmax`

Softmax 是一个常用的激活函数，通常用于多类别分类问题中，它将原始分数(也称为 logits)转换成归一化的概率分布。Softmax 函数的公式如下：

对于输入向量 $x$，Softmax 函数的输出 $y_i$ 计算如下：

$$ y_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} $$

其中，$N$ 是输入向量 $x$ 的维度，$x_i$ 是输入向量的第 $i$ 个元素。Softmax 函数会将每个元素 $x_i$ 转换为概率 $y_i$，使得所有 $y_i$ 的和等于1。

Softmax 函数的作用是将原始分数转换成概率分布，使得每个类别的概率值表示该类别的相对重要性。通常，在多类别分类问题中，模型的最后一层会使用 Softmax 函数将输出转换成概率分布，然后根据概率值选择最有可能的类别作为预测结果。

```python
import numpy as np

# 假设有一个输入向量 x
x = np.array([2.0, 1.0, 0.1])

# 计算 Softmax
softmax_x = np.exp(x) / np.sum(np.exp(x))

# 打印 Softmax 结果
print(softmax_x)
```

在深度学习框架 PyTorch 中，Softmax 通常作为激活函数在模型的最后一层使用，例如：

```python
import torch
import torch.nn.functional as F

# 假设有一个 PyTorch 张量 x
x = torch.tensor([2.0, 1.0, 0.1])

# 使用 PyTorch 的 F.softmax() 计算 Softmax
softmax_x = F.softmax(x, dim=0)

# 打印 Softmax 结果
print(softmax_x)
```

> 请注意，`dim` 参数用于指定 Softmax 在哪个维度上进行归一化，通常是在最后一个维度上进行操作。

## `cross_entropy`

交叉熵(Cross-Entropy)是在机器学习和深度学习中常用于衡量两个概率分布之间的差异的一种损失函数。在分类问题中，交叉熵通常用于衡量模型的预测分布与真实标签分布之间的差异。交叉熵损失越小，表示模型的预测越接近真实标签，反之亦然。

对于二分类问题，交叉熵损失函数的公式如下：

$$ H(y, p) = -(y \log(p) + (1 - y) \log(1 - p)) $$

其中：

- $H(y, p)$ 是交叉熵损失函数。
- $y$ 是真实标签(0或1)。
- $p$ 是模型的预测概率值(通常在0到1之间)。

对于多分类问题，交叉熵损失函数的公式如下：

$$ H(y,p) = - \sum_{i=1}^{N} y_i \log(p_i) $$

其中：

- $H(y, p)$ 是交叉熵损失函数。
- $N$ 是类别的数量。
- $y_i$ 是真实标签的独热编码(一个类别的标签为1，其他类别为0)。
- $p_i$ 是模型的预测概率分布。

在实际应用中，深度学习框架如 PyTorch 和 TensorFlow 提供了内置的交叉熵损失函数，用于计算损失并用于优化模型。在 PyTorch 中，你可以使用 `torch.nn.CrossEntropyLoss()` 来计算交叉熵损失，通常结合 Softmax 激活函数一起使用。

```python
import torch
import torch.nn as nn

# 创建一个示例的预测值和实际标签(多分类问题)
predicted = torch.tensor([[0.2, 0.7, 0.1], [0.9, 0.1, 0.0]])
actual = torch.tensor([2, 0])  # 真实标签，每个样本对应一个类别的索引

# 创建交叉熵损失函数实例
criterion = nn.CrossEntropyLoss()

# 计算交叉熵损失
loss = criterion(predicted, actual)

# 打印损失值
print(loss.item())
```

> 在上述示例中，我们使用了 `nn.CrossEntropyLoss()` 来计算交叉熵损失，传入模型的预测值 `predicted` 和真实标签 `actual`。最后，我们通过 `loss.item()` 获取损失值。