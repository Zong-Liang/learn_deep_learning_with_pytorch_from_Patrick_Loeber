# 12_feed_forward_net

## `nn.CrossEntropyLoss()`

`nn.CrossEntropyLoss()` 是 PyTorch 中用于计算交叉熵损失的损失函数类。交叉熵损失函数通常用于多类别分类问题中，它衡量了模型的预测分布与真实标签分布之间的差异。`nn.CrossEntropyLoss()` 的使用非常常见，特别是在神经网络的分类任务中。

以下是 `nn.CrossEntropyLoss()` 的一些重要参数和使用方法：

- `weight`(可选)：一个张量，用于指定每个类别的损失权重。默认值为 `None`，表示所有类别的权重都相等。
- `size_average`(可选)：一个布尔值，用于指定是否计算平均损失。如果设置为 `True`，则计算的是平均损失值；如果设置为 `False`，则计算的是总损失值。默认为 `True`。
- `ignore_index`(可选)：一个整数，用于指定忽略的标签索引。在计算损失时，将忽略具有此索引的标签。默认为 `-100`，表示不忽略任何标签。
- `reduction`(可选)：一个字符串，用于指定损失的缩减方式。可选值包括：
  - `'none'`：不进行缩减，返回每个样本的损失。
  - `'mean'`：计算平均损失值。
  - `'sum'`：计算总损失值。 默认为 `'mean'`。

使用 `nn.CrossEntropyLoss()` 的一般步骤如下：

1. 定义模型：构建一个神经网络模型，通常包括输入层、隐藏层和输出层。
2. 定义损失函数：创建一个 `nn.CrossEntropyLoss()` 类的实例，可以根据需要设置参数，如权重、平均方式等。
3. 准备数据：准备包含输入数据和真实标签的训练数据集。
4. 前向传播：使用模型进行前向传播，获取模型的预测结果。
5. 计算损失：将预测结果和真实标签传递给损失函数，计算损失值。
6. 反向传播和优化：根据损失值进行反向传播，更新模型的参数以最小化损失函数。

```python
import torch
import torch.nn as nn

# 示例的预测值和真实标签
predicted = torch.tensor([[0.2, 0.7, 0.1], [0.9, 0.1, 0.0]])
actual = torch.tensor([2, 0])  # 真实标签，每个样本对应一个类别的索引

# 创建交叉熵损失函数实例
criterion = nn.CrossEntropyLoss()

# 计算交叉熵损失
loss = criterion(predicted, actual)

# 打印损失值
print(loss.item())
```

> 在上述示例中，我们首先创建了一个 `nn.CrossEntropyLoss()` 实例，然后使用预测值和真实标签计算了交叉熵损失值。最后，通过 `loss.item()` 获取损失值。