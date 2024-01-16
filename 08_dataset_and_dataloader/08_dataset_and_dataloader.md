# 08_dataset_and_dataloader

## `from torch.utils.data import Dataset, DataLoader`

- `Dataset` 是 PyTorch 中的一个基类，用于自定义数据集。通过继承 `Dataset` 类，你可以创建自己的数据集类，以加载和处理你的数据。
- `DataLoader` 是 PyTorch 中的一个类，用于创建数据加载器，它可以方便地对数据集进行批量加载、洗牌和迭代。`DataLoader` 接受一个数据集对象(通常是继承自 `Dataset` 的自定义数据集类)，并根据指定的批量大小、是否随机洗牌等参数来生成数据批次。

这两个类通常一起使用，特别是在训练深度学习模型时。你可以使用自定义的 `Dataset` 类加载数据，然后使用 `DataLoader` 创建数据加载器，以便有效地训练模型。

```python
from torch.utils.data import Dataset, DataLoader

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self):
        # 初始化数据集

    def __getitem__(self, index):
        # 获取单个样本的数据和标签
        return data, label

    def __len__(self):
        # 返回数据集的长度

# 创建数据集实例
my_dataset = MyDataset()

# 创建数据加载器
batch_size = 64
data_loader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=True)

# 遍历数据加载器以访问批量数据
for data, label in data_loader:
    # 在这里进行训练或其他操作
    pass
```

## `torchvision.datasets`

`torchvision.datasets` 是 PyTorch 中的一个子模块，用于提供一些常见的计算机视觉数据集，以便于深度学习模型的训练和测试。这些数据集包括图像分类、目标检测、语义分割等领域的数据集。`torchvision.datasets` 模块的目的是让用户能够方便地下载、加载和使用这些标准数据集。

- MNIST: 手写数字识别数据集，包含了一系列0到9的手写数字图像。
- CIFAR-10 和 CIFAR-100: 包含了10个和100个不同类别的小图像的数据集，用于图像分类任务。
- ImageNet: 一个大规模图像数据集，包含数百万张图像，用于图像分类和其他计算机视觉任务。
- COCO (Common Objects in Context): 包含大量图像以及用于目标检测、语义分割等任务的注释信息。
- VOC (PASCAL Visual Object Classes): 包含多个对象类别的图像数据集，通常用于目标检测和语义分割。
- STL-10: 用于图像分类的数据集，包含10个类别的图像。

你可以使用 `torchvision.datasets` 模块中的函数来下载和加载这些数据集。

```python
import torchvision.datasets as datasets

# 创建一个 MNIST 数据集实例
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=None, download=True)

# 使用 DataLoader 进行数据加载
batch_size = 64
data_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

# 遍历数据加载器以访问批量数据
for images, labels in data_loader:
    # 在这里进行训练或其他操作
    pass
```

## `math.ceil()`

`math.ceil(x)` 是 Python 中 `math` 模块提供的一个函数，用于向上取整(向最接近正无穷大的整数方向取整数)。

具体来说，`math.ceil(x)` 接受一个浮点数 `x` 作为参数，并返回不小于 `x` 的最小整数。如果 `x` 已经是整数，则返回 `x` 本身。如果 `x` 是正数并且有小数部分，`math.ceil(x)` 将返回大于 `x` 的下一个整数。如果 `x` 是负数并且有小数部分，`math.ceil(x)` 将返回小于 `x` 的下一个整数。

```python
import math

x1 = 4.2
x2 = -3.7

ceil_x1 = math.ceil(x1)  # 结果为 5，向上取整到最接近的整数
ceil_x2 = math.ceil(x2)  # 结果为 -3，向上取整到最接近的整数

print(ceil_x1)
print(ceil_x2)
```