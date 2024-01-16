# 03_backpropagation

## `loss.backward()`

`loss.backward()` 是 PyTorch 中用于执行自动求导(反向传播)的方法。它通常用于神经网络的训练中，用于计算模型参数相对于损失函数的梯度，以便通过优化算法(如随机梯度下降)来更新模型的权重以最小化损失。

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
```

> `loss.backward()` 方法会自动计算梯度并使用链式法则将梯度传播回模型的参数。这使得在深度学习中可以高效地计算梯度，并用于更新模型参数以最小化损失函数。
>
> 需要注意的是，`backward()` 方法通常在每个训练批次之后调用，以计算当前批次的梯度信息，然后可以使用这些梯度来更新模型参数。