# 01_tensor_basics

## `torch.empty()`

`torch.empty()`用于创建一个未初始化的张量(tensor)，你可以在创建时指定张量的形状(shape)。

```python
import torch

# 创建一个未初始化的1维张量(长度为5)
empty_tensor = torch.empty(5)
print(empty_tensor)

# 创建一个未初始化的2x3的二维张量
empty_matrix = torch.empty(2, 3)
print(empty_matrix)
```

> 请注意，`torch.empty` 创建的张量不会被初始化为特定的数值，它们的内容可能是随机的或者是之前在内存中存在的值。

## `torch.rand()` 

`torch.rand()` 用于创建一个包含随机数值的张量(tensor)。你可以指定张量的形状(shape)来创建不同维度的随机数值张量。

```python
import torch

# 创建一个包含随机数值的1维张量(长度为5)
random_tensor = torch.rand(5)
print(random_tensor)

# 创建一个包含随机数值的2x3的二维张量
random_matrix = torch.rand(2, 3)
print(random_matrix)
```

> 默认情况下，`torch.rand` 会生成位于 `[0, 1)` 范围内的均匀分布的随机数值。如果你需要生成不同的分布，可以使用其他函数，例如 `torch.randn` 用于生成标准正态分布的随机数值，或者 `torch.randint` 用于生成整数随机数值。

## `torch.zeros()`

`torch.zeros()` 用于创建一个全零的张量(tensor)。你可以指定张量的形状(shape)来创建不同维度的全零张量。

```python
import torch

# 创建一个全零的1维张量(长度为5)
zeros_tensor = torch.zeros(5)
print(zeros_tensor)

# 创建一个全零的2x3的二维张量
zeros_matrix = torch.zeros(2, 3)
print(zeros_matrix)
```

> `torch.zeros()` 创建的张量的所有元素都被初始化为零。这在初始化神经网络的权重矩阵等任务中非常有用。

## `torch.ones()` 

`torch.ones()`用于创建一个全为1的张量(tensor)。你可以指定张量的形状(shape)来创建不同维度的全1张量。

```python
import torch

# 创建一个全1的1维张量(长度为5)
ones_tensor = torch.ones(5)
print(ones_tensor)

# 创建一个全1的2x3的二维张量
ones_matrix = torch.ones(2, 3)
print(ones_matrix)
```

> `torch.ones()` 创建的张量的所有元素都被初始化为1。这在初始化神经网络的偏置(bias)矩阵等任务中非常有用。

`torch.tensor()` 用于创建张量(tensor)。与其他创建张量的函数不同，`torch.tensor()` 允许你从已有的数据(例如 Python 列表、NumPy 数组等)创建张量，并且可以指定数据类型(dtype)和设备(device)。

```python
import torch

# 从Python列表创建张量
my_list = [1, 2, 3, 4, 5]
my_tensor = torch.tensor(my_list)
print(my_tensor)

# 从Python列表创建张量，并指定数据类型为浮点数
my_float_tensor = torch.tensor(my_list, dtype=torch.float32)
print(my_float_tensor)

# 从Python列表创建张量，并将其放在GPU上(如果可用)
my_gpu_tensor = torch.tensor(my_list, device='cuda')
print(my_gpu_tensor)

```

> `torch.tensor()` 具有灵活性，因为它允许你从各种数据源创建张量，并且可以指定数据类型和设备，以满足你的需求。这在将外部数据导入到 PyTorch 中时非常有用。

## `torch.add()`

`torch.add(input, other)`: 执行张量相加操作。

```python
import torch

# 创建两个张量
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

# 执行相加操作
result = torch.add(tensor1, tensor2)
result = tensor1.add_(tensor2)
print(result)  # 输出: tensor([5., 7., 9.])
```

## `torch.sub()`

`torch.sub(input, other)`: 执行张量相减操作。

```python
import torch

# 创建两个张量
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

# 执行相减操作
result = torch.sub(tensor1, tensor2)
result = tensor1.sub_(tensor2)
print(result)  # 输出: tensor([-3., -3., -3.])
```

## `torch.mul()`

`torch.mul(input, other)`: 执行张量相乘操作。

```python
import torch

# 创建两个张量
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

# 执行相乘操作
result = torch.mul(tensor1, tensor2)
result = tensor1.mul_(tensor2)
print(result)  # 输出: tensor([ 4., 10., 18.])
```

## `torch.div()`

`torch.div(input, other)`: 执行张量相除操作。

```python
import torch

# 创建两个张量
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

# 执行相除操作
result = torch.div(tensor1, tensor2)
result = tensor1.div_(tensor2)
print(result)  # 输出: tensor([0.2500, 0.4000, 0.5000])
```

> 这些函数允许你执行元素级别的操作，即将一个张量中的每个元素与另一个张量中的对应元素进行操作。

## `x.item() `

`x.item()` 用于获取一个标量张量(scalar tensor)中的单个数值。标量张量是只包含一个元素的张量，通常用于表示单一的数值结果，例如损失函数的结果或模型的输出。

```python
import torch

# 创建一个标量张量
scalar_tensor = torch.tensor(42.0)

# 使用 item() 方法获取标量张量中的数值
value = scalar_tensor.item()
print(value)  # 输出：42.0
```

> 需要注意的是，`item()` 方法仅适用于标量张量，即只包含一个元素的张量。如果你尝试在包含多个元素的张量上使用 `item()`，将会引发错误。因此，确保在使用 `item()` 之前检查张量的形状。

## `x.view()`

`x.view()` 用于改变张量(tensor)形状。它允许你重新组织张量中的元素，以创建具有不同形状的新张量，但不改变元素的数量。

```python
import torch

# 创建一个形状为(2, 3)的二维张量
original_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 使用 view 方法将其变换为形状为(3, 2)的新二维张量
reshaped_tensor = original_tensor.view(3, 2)

print(original_tensor)
print(reshaped_tensor)
```

> 需要注意的是，`view()` 方法不会复制张量的数据，它只是通过重新排列元素的方式创建一个新的张量视图。因此，原始张量和重新形状的张量共享相同的数据存储，如果你修改其中一个张量的内容，另一个张量也会受到影响。
>
> 另外，要确保重新形状的操作是有效的，即新形状的元素数量必须与原始形状的元素数量一致。否则，将会引发错误。

## `x.numpy()`

`x.numpy()` 用于将张量(tensor)转换为 NumPy 数组。这个方法可以让你在 PyTorch 和 NumPy 之间进行数据的无缝转换，因为这两个库通常一起使用。

```python
import torch
import numpy as np

# 创建一个 PyTorch 张量
torch_tensor = torch.tensor([1.0, 2.0, 3.0])

# 使用 numpy() 方法将 PyTorch 张量转换为 NumPy 数组
numpy_array = torch_tensor.numpy()

# 现在 numpy_array 是一个 NumPy 数组
print(numpy_array)
```

> 需要注意的是，`numpy()` 方法返回的 NumPy 数组与原始 PyTorch 张量共享相同的数据存储，因此对其中一个数据结构的修改会影响另一个。这在某些情况下可能会导致不期望的结果，因此要小心处理。
>
> 另外，如果 PyTorch 张量位于 GPU 上，使用 `numpy()` 方法将其转换为 NumPy 数组时会将数据从 GPU 移回 CPU，因此需要确保在进行转换之前已经将数据移到 CPU 上。

## `torch.from_numpy()`

`torch.from_numpy()` 是 PyTorch 中的一个函数，用于将 NumPy 数组转换为 PyTorch 张量(tensor)。

```python
import numpy as np
import torch

# 创建一个 NumPy 数组
numpy_array = np.array([1.0, 2.0, 3.0])

# 使用 torch.from_numpy() 将 NumPy 数组转换为 PyTorch 张量
torch_tensor = torch.from_numpy(numpy_array)

# 现在 torch_tensor 是一个 PyTorch 张量
print(torch_tensor)
```

> 需要注意的是，`torch.from_numpy()` 创建的 PyTorch 张量与原始 NumPy 数组共享相同的数据存储，因此对其中一个数据结构的修改会影响另一个。这在某些情况下可能会导致不期望的结果，因此要小心处理。
>
> 另外，如果你希望将 NumPy 数组转换为 GPU 上的 PyTorch 张量，需要在转换后使用 `.to('cuda')` 或 `.cuda()` 方法将其移到 GPU 上。

## `torch.cuda.is_available()`

`torch.cuda.is_available()`用于检查当前系统是否支持 GPU 并且是否安装了支持 CUDA 的 PyTorch 版本。CUDA 是 NVIDIA 提供的并行计算平台，用于在 GPU 上加速深度学习任务。

这个函数返回一个布尔值，如果系统支持 CUDA 并且已经安装了支持 CUDA 的 PyTorch 版本，则返回 `True`，否则返回 `False`。

```python
import torch

# 检查系统是否支持 CUDA
if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")
    
# 创建一个 PyTorch 张量并将其移到 GPU 上
tensor = torch.tensor([1.0, 2.0, 3.0])
if torch.cuda.is_available():
    tensor = tensor.cuda()
```

## `torch.device()`

`torch.device()`用于创建一个代表设备的对象，通常用于在 CPU 和 GPU 之间切换张量(tensor)的计算设备。它允许你明确指定在哪个设备上执行 PyTorch 操作。

```python
import torch

# 创建一个代表 CPU 设备的对象
cpu_device = torch.device("cpu")

# 创建一个代表 GPU 设备的对象，可以使用 GPU 的索引来指定(例如：0代表第一个GPU)
gpu_device = torch.device("cuda:0")

# 创建一个 PyTorch 张量，并将其移到指定的设备上
tensor = torch.tensor([1.0, 2.0, 3.0], device=gpu_device)

# 打印张量所在的设备
print(tensor.device)
```

> 使用 `torch.device()` 可以在多 GPU 环境中方便地选择在哪个 GPU 上执行计算，或者在没有 GPU 的情况下将计算移回 CPU。此外，当你在使用多 GPU 时，还可以使用`torch.nn.DataParallel` 将模型在多个 GPU 上并行运行。

##  `torch.nn.DataParallel()`

`torch.nn.DataParallel` 是 PyTorch 中的一个模块，用于在多个 GPU 上并行运行模型。它可以将一个模型复制到多个 GPU 上，并自动处理输入数据的分发以及梯度的合并，从而实现模型的并行计算。

```python
import torch
import torch.nn as nn

# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建一个模型实例
model = SimpleModel()

# 如果系统支持多个 GPU
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

# 将模型移到 GPU(如果有的话)
if torch.cuda.is_available():
    model = model.cuda()

# 创建一个示例输入张量(batch_size=4, 特征数=10)
input_data = torch.randn(4, 10)

# 将输入数据移动到 GPU(如果模型在 GPU 上)
if torch.cuda.is_available():
    input_data = input_data.cuda()

# 使用模型进行前向传播
output = model(input_data)

# 打印输出
print(output)
```

> `DataParallel` 模块会自动将输入数据分发到多个 GPU 上，并在计算结束后将梯度合并回主 GPU。这样，你可以更轻松地利用多个 GPU 进行模型的训练，从而提高训练速度。