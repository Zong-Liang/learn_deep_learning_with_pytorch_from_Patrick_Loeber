# 16_save_and_load_models

## `torch.save()`

`torch.save()` 是 PyTorch 中用于保存模型、张量和其他 Python 对象到文件的函数。它允许你将 PyTorch 模型的状态字典、张量和其他对象保存到磁盘上，以便以后加载和使用。

以下是 `torch.save()` 的基本用法和参数：

- `obj`：要保存的对象，通常是模型的状态字典、张量或其他 Python 对象。
- `f`：保存的文件路径，可以是字符串表示的文件名或一个文件对象(如 Python 的 `io.BytesIO`)。
- `pickle_module`：指定用于序列化对象的模块。默认情况下，PyTorch 使用 Python 的 `pickle` 模块。你可以指定其他序列化模块，如 `pickle_module=cloudpickle`。
- `pickle_protocol`：指定用于序列化的协议版本。默认情况下，使用 Python 的默认协议版本。你可以指定其他协议版本，如 `pickle_protocol=4`。
- `_use_new_zipfile_serialization`：一个内部参数，通常无需手动设置。

## `torch.load()`

`torch.load()` 是 PyTorch 中的一个函数，用于从磁盘上加载保存的模型、张量或其他 Python 对象。它的作用是将之前使用 `torch.save()` 保存的对象加载回内存中，以便在训练、评估或部署中使用。

以下是 `torch.load()` 的基本用法和参数：

- `f`：加载的文件路径，可以是字符串表示的文件名或一个文件对象(如 Python 的 `io.BytesIO`)。
- `map_location`：指定加载的对象在哪个设备上运行。默认情况下，加载的对象将放置在 CPU 上，但你可以指定一个设备，如 `map_location='cuda:0'`，以将对象加载到指定的 GPU 上。
- `pickle_module`：指定用于反序列化对象的模块。默认情况下，PyTorch 使用 Python 的 `pickle` 模块。你可以指定其他反序列化模块，如 `pickle_module=cloudpickle`。

## `load_state_dict()`

`load_state_dict()` 方法是 PyTorch 中 `nn.Module` 类的一个方法，用于加载模型的状态字典(state dictionary)。它的作用是将事先保存好的模型参数(权重和偏置等)从状态字典加载到模型中，以便在训练或推理时使用。

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
    obj=model.state_dict(),  # only saving the state_dict() - only saves the models learned parameters
    f=MODEL_SAVE_PATH,
)
```

