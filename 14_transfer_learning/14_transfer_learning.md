# 14_transfer_learning

## `torchvision.utils.make_grid()`

`torchvision.utils.make_grid()` 是 PyTorch 中的一个函数，用于将多个图像张量合并成一个网格图像。通常，它用于可视化深度学习模型的输出、输入或中间层特征图等。

以下是 `torchvision.utils.make_grid()` 的一些重要参数和作用：

- `tensor`：一个包含多个图像张量的张量或列表。每个图像张量通常是一个四维张量，其形状为 `(batch_size, channels, height, width)`。
- `nrow`：可选参数，表示网格中每行的图像数量。默认值为 8。
- `padding`：可选参数，表示每个图像之间的填充像素数。默认值为 2。
- `normalize`：可选参数，如果设置为 `True`，则对图像进行归一化，使其像素值在 [0, 1] 范围内。
- `range`：可选参数，用于指定图像像素值的范围，格式为 `(min, max)`。
- `scale_each`：可选参数，如果设置为 `True`，则对每个图像单独进行归一化。
- `pad_value`：可选参数，用于指定填充像素的值，默认为 0。

`torchvision.utils.make_grid()` 将多个图像张量合并成一个大的图像张量，这个大图像包含了网格中的所有图像。通常，它用于可视化模型的输入、输出、中间层特征等，以便更容易观察和理解模型的行为。

```python
import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# 创建一个包含多个图像张量的示例列表
image_tensors = [torch.randn(3, 224, 224) for _ in range(16)]

# 将图像张量合并成一个网格图像
grid_image = make_grid(image_tensors, nrow=4, padding=10)

# 将网格图像转换为 NumPy 数组
grid_image_numpy = grid_image.permute(1, 2, 0).numpy()

# 显示网格图像
plt.imshow(grid_image_numpy)
plt.axis('off')
plt.show()
```

> 在上述示例中，我们首先创建了一个包含多个示例图像张量的列表 `image_tensors`。然后，使用 `torchvision.utils.make_grid()` 将这些图像张量合并成一个网格图像 `grid_image`，并将其显示出来。

## `lr_scheduler.StepLR()`

`lr_scheduler.StepLR()` 是 PyTorch 中的一个学习率调度器(LR scheduler)类，用于在训练深度学习模型时调整学习率的策略之一。`StepLR` 调度器的作用是在训练的特定时期(或特定的训练周期)降低学习率。

以下是 `lr_scheduler.StepLR()` 的一些重要参数和作用：

- `optimizer`：需要调整学习率的优化器，通常是使用的优化器的实例，例如 `torch.optim.SGD` 或 `torch.optim.Adam`。
- `step_size`：表示学习率下降的周期，即每经过 `step_size` 个训练周期后，学习率会降低。
- `gamma`：表示学习率下降的倍数，即学习率降低的因子。通常设置为小于 1 的值，例如 `gamma=0.1` 表示学习率每次下降为原来的 0.1 倍。

`lr_scheduler.StepLR()` 通过监控训练周期的数量来降低学习率，当达到 `step_size` 的整数倍时，学习率会乘以 `gamma`。这种策略通常用于训练深度学习模型，以确保随着训练的进行，学习率逐渐减小，有助于模型更好地收敛。

```python
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 创建一个优化器(例如，SGD)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 创建一个 StepLR 学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 在训练循环中使用调度器
for epoch in range(100):
    # 在每个训练周期之前更新学习率
    scheduler.step()
    
    # 训练模型
    train(model, train_loader)
```

> 在上述示例中，我们首先创建了一个优化器 `optimizer`，然后创建了一个 `StepLR` 学习率调度器 `scheduler`，设置 `step_size` 为 10，`gamma` 为 0.1。在训练循环中，我们在每个训练周期之前使用 `scheduler.step()` 来更新学习率。这将确保在经过每 10 个训练周期后，学习率降低为原来的 0.1 倍，以帮助模型更好地训练。

## `transfer learning`

```python
#### ConvNet as fixed feature extractor ####
# Here, we need to freeze all the network except the final layer.
# We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
print(model_conv)
```

```python
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=2, bias=True)
)
```

