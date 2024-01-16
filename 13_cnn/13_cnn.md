# 13_cnn

## `nn.Conv2d()`

`nn.Conv2d()` 是 PyTorch 中用于定义二维卷积层的类。卷积层是深度神经网络中的关键组件，用于从输入数据中提取特征。`nn.Conv2d()` 类允许你创建二维卷积层，并可以自定义卷积核、步幅、填充等参数。

以下是 `nn.Conv2d()` 的一些重要参数和使用方法：

- `in_channels`：输入通道数，即输入数据的通道数。例如，对于 RGB 图像，通常为 3。
- `out_channels`：输出通道数，即卷积层的卷积核数量，每个卷积核产生一个输出通道。
- `kernel_size`：卷积核的大小，可以是整数或元组，例如 `(3, 3)` 表示 3x3 大小的卷积核。
- `stride`：卷积核的步幅，控制滑动窗口的移动距离。
- `padding`：填充，可以是 0(不填充)或非负整数，控制输入数据的边界填充。
- `dilation`：膨胀率，用于空洞卷积。默认值为 1。
- `groups`：分组卷积的组数，默认值为 1。分组卷积是一种特殊的卷积操作，将输入数据和卷积核分成多个组进行卷积操作。
- `bias`：一个布尔值，指定是否包含偏置项。默认为 `True`，表示包含偏置项。
- `padding_mode`：填充模式，可以是 'zeros'(默认值)或 'reflect'。
- `padding_mode`：填充模式，可以是 'zeros'(默认值)或 'reflect'。
- `padding_mode`：填充模式，可以是 'zeros'(默认值)或 'reflect'。

使用 `nn.Conv2d()` 创建卷积层的一般步骤如下：

1. 导入 PyTorch 库：首先，需要导入 PyTorch 库。
2. 创建卷积层实例：使用 `nn.Conv2d()` 创建卷积层的实例，并指定相关参数。
3. 前向传播：将输入数据传递给卷积层实例进行前向传播，从而生成输出。

```python
import torch
import torch.nn as nn

# 创建一个卷积层实例
conv_layer = nn.Conv2d(
    in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
)

# 构造一个输入张量(例如，3通道的28x28图像)
input_data = torch.randn(1, 3, 28, 28)

# 将输入数据传递给卷积层进行前向传播
output = conv_layer(input_data)

# 打印输出的形状
print(output.shape)  # torch.Size([1, 64, 28, 28])
```

## `nn.MaxPool2d()`

`nn.MaxPool2d()` 是 PyTorch 中用于定义二维最大池化层的类。最大池化是一种常用的池化操作，通常用于减小特征图的尺寸并减少计算量，同时保留图像中的重要特征。

以下是 `nn.MaxPool2d()` 的一些重要参数和使用方法：

- `kernel_size`：池化窗口的大小，可以是整数或元组，例如 `(2, 2)` 表示 2x2 大小的池化窗口。
- `stride`：池化操作的步幅，控制滑动窗口的移动距离。
- `padding`：填充，可以是 0(不填充)或非负整数，控制输入数据的边界填充。
- `dilation`：膨胀率，用于空洞池化。默认值为 1。
- `return_indices`：一个布尔值，用于指定是否返回池化操作的索引。默认为 `False`，表示不返回索引。
- `ceil_mode`：一个布尔值，用于指定是否使用上取整的池化操作。默认为 `False`，表示使用下取整。

使用 `nn.MaxPool2d()` 创建最大池化层的一般步骤如下：

1. 导入 PyTorch 库：首先，需要导入 PyTorch 库。
2. 创建池化层实例：使用 `nn.MaxPool2d()` 创建池化层的实例，并指定相关参数。
3. 前向传播：将输入数据传递给池化层实例进行前向传播，从而生成池化后的输出。

```python
import torch
import torch.nn as nn

# 创建一个最大池化层实例
maxpool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

# 构造一个输入张量(例如，3通道的28x28图像)
input_data = torch.randn(1, 3, 28, 28)

# 将输入数据传递给最大池化层进行前向传播
output = maxpool_layer(input_data)

# 打印输出的形状
print(output.shape)  # torch.Size([1, 3, 14, 14])
```

## `torch.save()`

```python
# 定义了保存模型的文件夹路径
MODEL_PATH = Path("checkpoints")
# 用于创建模型保存文件夹
# parents=True 表示如果文件夹的上级目录也不存在，也会创建上级目录。
# exist_ok=True 表示如果文件夹已经存在，不会引发错误。
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 定义了保存的模型文件名
MODEL_NAME = "cnn.pth"
# 使用 / 运算符将模型文件的路径连接到模型保存文件夹的路径
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 打印保存模型的文件路径
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(
    obj=model.state_dict(),  # only saving the models learned parameters
    f=MODEL_SAVE_PATH,
)
```

