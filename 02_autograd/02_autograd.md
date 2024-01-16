# 02_autograd

## `torch.manual_seed()`

`torch.manual_seed()` 用于设置随机数生成器的种子(seed)，从而使随机操作在每次运行时具有确定性。通过设置随机数种子，你可以确保在相同种子下的随机操作在不同的运行中生成相同的随机数，从而使实验可重复。

```python
import torch

# 设置随机数种子
torch.manual_seed(42)

# 创建一个随机张量
random_tensor = torch.rand(3, 3)
print(random_tensor)
```

> 通过设置随机数种子，你可以确保在进行实验或开发时获得可重复的结果，这在深度学习中非常有用，特别是在调试和比较不同模型或算法时。但要注意，如果你在代码的其他部分使用了不同的种子，那么随机数生成器将不再具有确定性。

## `torch.randn()`

`torch.randn()` 用于创建随机张量(tensor)，其中的元素是从标准正态分布(均值为0，标准差为1)中抽取的随机数。你可以指定张量的形状来创建不同维度的随机张量。

```python
import torch

# 创建一个形状为(3, 3)的随机张量
random_tensor = torch.randn(3, 3)
print(random_tensor)

# 创建一个形状为(5,)的一维随机张量
random_tensor_1d = torch.randn(5)

# 创建一个形状为(2, 4)的二维随机张量
random_tensor_2d = torch.randn(2, 4)

# 创建一个形状为(3, 2, 2)的三维随机张量
random_tensor_3d = torch.randn(3, 2, 2)
```

> `torch.randn()` 通常用于初始化神经网络的权重(参数)或创建随机噪声张量，它是深度学习中常用的工具之一。

## `x.mean()`

`x.mean()` 用于计算张量(tensor)平均值。它返回张量中所有元素的平均值。

```python
import torch

# 创建一个示例张量
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 计算张量的平均值
mean_value = tensor.mean()
print(mean_value)

# 创建一个二维张量
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

# 计算每一列的平均值(沿第0维度)
column_means = matrix.mean(dim=0)
print(column_means)
```

> `x.mean()` 的灵活性允许你计算张量的整体平均值或沿特定维度的平均值，这在数据处理和深度学习中非常有用。

## `x.backward()`

`x.backward()` 用于自动求导(反向传播)。它通常用于神经网络的训练中，用于计算模型参数相对于损失函数的梯度，从而更新模型的权重以最小化损失。

```python
import torch

# 创建一个张量
x = torch.tensor(2.0, requires_grad=True)

# 定义一个损失函数(示例中为平方损失)
loss = (x - 4)**2

# 使用反向传播计算梯度
loss.backward()

# 打印 x 相对于损失函数的梯度
print(x.grad)
```

> `backward()` 方法将根据计算图自动计算梯度，然后将梯度传播回模型的参数，允许你使用优化算法(如随机梯度下降)来更新模型参数，以最小化损失函数。

## `x.detach()`

`x.detach()` 用于创建一个与原始张量(tensor)共享数据存储的新张量，但新张量没有梯度信息。这意味着使用 `detach()` 方法的张量将不再被视为计算图的一部分，不会跟踪任何梯度信息，因此在某些情况下可以用于分离梯度计算。

```python
import torch

# 创建一个张量并设置 requires_grad=True，以跟踪梯度
x = torch.tensor(2.0, requires_grad=True)

# 定义一个损失函数(示例中为平方损失)
loss = (x - 4)**2

# 使用反向传播计算梯度
loss.backward()

# 创建一个分离梯度信息的新张量
detached_x = x.detach()

# 打印 detached_x 是否需要梯度信息
print(detached_x.requires_grad)  # 输出：False
```

> 通常，`detach()` 方法在需要分离梯度信息以进行一些操作(如可视化、保存模型或进行推理)时很有用。这允许你在不改变原始张量的情况下，创建一个不需要梯度信息的新张量。

## `with torch.no_grad()`

`with torch.no_grad()` 用于指示 PyTorch 不要跟踪在其内部进行的张量操作的梯度信息。这在某些情况下非常有用，特别是在评估模型或进行推理时，不需要计算梯度。

```python
import torch

# 创建一个张量并设置 requires_grad=True，以跟踪梯度
x = torch.tensor(2.0, requires_grad=True)

# 在 torch.no_grad() 上下文中执行操作，不会跟踪梯度
with torch.no_grad():
    y = x * 3

# 打印 y 是否需要梯度信息
print(y.requires_grad)  # 输出：False
```

> `with torch.no_grad()` 主要用于在不需要计算梯度的情况下执行一些操作，例如在评估模型、进行推理或保存模型参数时。通过将操作包装在这个上下文中，可以提高代码的执行效率并减少不必要的梯度计算。

## `x.grad.zero_()`

`x.grad.zero_()` 用于将张量(tensor)的梯度信息清零。这个方法通常在训练循环中用于每个批次之后或每次优化步骤之后清除梯度，以便进行下一轮的梯度累积。

```python
import torch

# 创建一个张量并设置 requires_grad=True，以跟踪梯度
x = torch.tensor(2.0, requires_grad=True)

# 定义一个损失函数(示例中为平方损失)
loss = (x - 4)**2

# 使用反向传播计算梯度
loss.backward()

# 打印 x 相对于损失函数的梯度
print(x.grad)

# 清零梯度
x.grad.zero_()

# 打印清零后的梯度
print(x.grad)
```

> 清零梯度通常在每个训练批次之后或每次优化步骤之后使用，以确保新的梯度信息不会与之前的梯度信息累积。这在深度学习训练中非常常见，因为通常需要多次迭代来更新模型参数。