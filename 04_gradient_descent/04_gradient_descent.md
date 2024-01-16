# 04_gradient_descent

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