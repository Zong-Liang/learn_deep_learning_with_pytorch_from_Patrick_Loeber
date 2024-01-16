# 06_linear_regression

## `datasets.make_regression()`

`datasets.make_regression()` 是 scikit-learn 中的一个函数，用于生成一个回归任务的人工数据集。这个函数可以用来创建用于回归问题的合成数据集，其中包括特征和目标变量。

- `n_samples`: 数据集中的样本数量，默认为100。
- `n_features`: 数据集中的特征数量，默认为1。
- `noise`: 添加到目标变量中的噪声标准差，默认为20。噪声用于模拟目标变量中的随机性。
- `random_state`: 随机种子，用于控制生成的随机数据的可重复性。

```python
from sklearn.datasets import make_regression

# 创建一个回归任务的合成数据集
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# X 是特征矩阵，y 是目标变量
print("特征矩阵 X:\n", X)
print("目标变量 y:\n", y)
```

> 这个函数通常用于生成用于回归模型的训练数据，以便进行回归问题的实验和测试。你可以根据需要调整参数来创建不同规模和性质的合成数据集。

## `nn.Linear`

`nn.Linear` 是 PyTorch 中的一个模块，用于创建线性(全连接)层，通常用于神经网络中。线性层是神经网络中的基本层之一，它执行线性变换，将输入特征映射到输出特征。

- `in_features`: 输入特征的数量。

- `out_features`: 输出特征的数量。

```python
import torch
import torch.nn as nn

# 创建一个线性层，输入特征数为3，输出特征数为2
linear_layer = nn.Linear(in_features=3, out_features=2)

# 创建一个示例输入张量(batch_size=4，特征数=3)
input_data = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0],
                           [10.0, 11.0, 12.0]])

# 使用线性层进行前向传播
output = linear_layer(input_data)

# 打印输出
print(output)
```

> 性层通常用于神经网络中的线性变换，是神经网络中的一个重要组件。它将输入特征与权重矩阵相乘，并添加偏置，从而将输入数据映射到输出特征空间。这些权重和偏置参数会在训练过程中学习，以使模型适应数据。

## `nn.MSELoss()`

`nn.MSELoss()` 是 PyTorch 中的一个损失函数，用于计算均方误差(Mean Squared Error，MSE)。均方误差是回归问题中常用的损失函数之一，用于衡量模型的预测值与实际值之间的差异。

MSE 的计算方式如下：

$MSE = (1/n) * Σ(predicted - actual)^2$

其中，n 是样本数量，predicted 是模型的预测值，actual 是实际值。MSE 的值越小，表示模型的预测越接近实际值。

```python
import torch
import torch.nn as nn

# 创建一个示例的预测值和实际值
predicted = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
actual = torch.tensor([2.0, 2.5, 3.5], requires_grad=False)

# 创建一个 MSE 损失函数实例
criterion = nn.MSELoss()

# 计算均方误差
loss = criterion(predicted, actual)

# 打印损失值
print(loss)
```

> MSE 常用于回归问题的损失函数，它衡量了模型的预测与真实标签之间的差异。通常，训练过程的目标是最小化 MSE，以使预测值更接近真实值。

## `torch.optim.SGD()`

`torch.optim.SGD()` 是 PyTorch 中的一个优化器，用于实现随机梯度下降(Stochastic Gradient Descent，SGD)算法，是深度学习中常用的优化算法之一。SGD 用于更新神经网络的参数以最小化损失函数。

- `params`: 一个包含模型参数的迭代器(通常是通过 `model.parameters()` 获取的)。
- `lr`(学习率): 一个控制梯度更新步长的正数值。学习率越小，参数更新越稳定，但收敛速度较慢；学习率越大，参数更新速度较快，但可能会不稳定。
- `momentum`(动量): 一个介于0到1之间的正数值，用于加速参数更新过程。动量可以帮助模型跳出局部极小值，通常设置为0.9。
- `dampening`: 一个介于0到1之间的正数值，用于控制动量的抑制(阻尼)。
- `weight_decay`(权重衰减): 一个正数值，用于在梯度更新中应用 L2 正则化，以减小参数的大小。它有助于防止过拟合。
- `nesterov`(Nesterov 动量): 一个布尔值，控制是否使用 Nesterov 动量。如果设置为 `True`，则使用 Nesterov 动量；如果设置为 `False`，则使用标准动量。

```python
import torch
import torch.optim as optim

# 创建一个模型实例
model = torch.nn.Linear(2, 1)

# 创建一个 SGD 优化器，设置学习率和动量
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

# 在训练循环中使用优化器进行参数更新
for epoch in range(epochs):
    optimizer.zero_grad()  # 清零梯度
    outputs = model(inputs)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数
```

> SGD 优化器是深度学习中常用的优化算法之一，通过迭代优化模型参数，使其逐渐逼近损失函数的最小值，从而提高模型的性能。