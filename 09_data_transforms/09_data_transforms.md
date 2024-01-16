# 09_data_transforms

## `torchvision.transforms.Compose()`

`torchvision.transforms.Compose(transforms)` 是 PyTorch 中 torchvision 库中的一个函数，用于将多个数据转换操作组合成一个数据转换管道。这个函数通常用于对图像数据进行数据预处理、数据增强和数据转换。

- `transforms`(列表或元组)：包含多个数据转换操作的列表或元组。这些数据转换操作按顺序应用于输入数据，并返回转换后的数据。通常，每个数据转换操作都是一个类对象，它具有 `__call__` 方法，以便在转换管道中被调用。

例如，假设你有三个不同的数据转换操作 `transform1`、`transform2` 和 `transform3`，你可以使用 `torchvision.transforms.Compose()` 将它们组合成一个转换管道：

```python
from torchvision.transforms import Compose

transforms_pipeline = Compose([transform1, transform2, transform3])

transformed_data = transforms_pipeline(original_data)
```

> 这种方式非常有用，特别是在图像处理中，你可以将各种数据增强、预处理和转换操作组合在一起，以便在训练深度学习模型时方便地对数据进行处理。