# 07_logistic_regression

## `datasets.load_breast_cancer()`

`datasets.load_breast_cancer()` 是 scikit-learn(sklearn)库中的一个函数，用于加载乳腺癌数据集。这个数据集包含了一些乳腺癌肿瘤样本的特征和标签，用于机器学习任务。

```python
from sklearn import datasets

breast_cancer_data = datasets.load_breast_cancer()
```

- `breast_cancer_data.data` 包含了特征数据，是一个二维数组，每一行代表一个样本，每一列代表一个特征。
- `breast_cancer_data.target` 包含了标签数据，是一个一维数组，每个元素对应一个样本的类别标签(0 表示恶性肿瘤，1 表示良性肿瘤)。
- `breast_cancer_data.feature_names` 包含了特征的名称。
- `breast_cancer_data.target_names` 包含了类别的名称

> 你可以根据你的机器学习任务，使用这些数据进行模型的训练、测试和评估。例如，你可以使用这个数据集来训练一个分类模型来预测肿瘤是恶性还是良性。

## `train_test_split()`

`train_test_split()` 是 scikit-learn(sklearn)库中一个用于将数据集分割成训练集和测试集的函数，通常用于机器学习模型的训练和评估。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 加载乳腺癌数据集
breast_cancer_data = load_breast_cancer()

# 获取特征数据和标签数据
X = breast_cancer_data.data  # 特征数据
y = breast_cancer_data.target  # 标签数据

# 分割数据集，指定测试集占总数据的比例(这里是80%训练，20%测试)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

> 一旦你完成了数据集的分割，你就可以使用训练集来训练机器学习模型，然后使用测试集来评估模型的性能。

## `StandardScaler()`

`StandardScaler` 是 scikit-learn(sklearn)库中的一个用于特征标准化的类。特征标准化是数据预处理的一种常见技术，用于将不同特征的值缩放到相同的尺度，以确保它们对机器学习模型的训练产生均衡的影响。`StandardScaler` 类通过将特征的均值调整为0，标准差调整为1来实现标准化。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# 加载乳腺癌数据集
breast_cancer_data = load_breast_cancer()

# 获取特征数据
X = breast_cancer_data.data  # 特征数据

# 创建一个 StandardScaler 实例
scaler = StandardScaler()

# 使用 StandardScaler 对特征数据进行标准化
X_scaled = scaler.fit_transform(X)
```

> 一旦你完成了特征标准化，你可以将标准化后的数据用于机器学习模型的训练。标准化有助于确保不同特征的值具有相同的尺度，从而改善模型的性能。
>
> 请注意，标准化通常是在训练模型之前进行的预处理步骤，以确保模型的稳定性和准确性。在应用模型时，也需要对新的数据进行相同的标准化操作，以保持一致性。

## `nn.BCELoss()`

`nn.BCELoss()` 是 PyTorch 中的一个损失函数，用于二分类问题中的二元交叉熵损失(Binary Cross-Entropy Loss)。它通常用于衡量二分类模型的预测与真实标签之间的差异。

对于二元交叉熵损失，它的计算方式如下：

$BCE Loss = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]$

其中，y 是真实标签(0或1)，y_pred 是模型的预测概率值(通常在0到1之间)。BCE Loss 的值越小，表示模型的预测越接近真实标签。

```python
import torch
import torch.nn as nn

# 创建一个示例的预测值和实际标签
predicted = torch.tensor([0.7, 0.3, 0.8], requires_grad=True)  # 模型的预测概率
actual = torch.tensor([1, 0, 1], dtype=torch.float32)  # 真实标签，通常是0或1

# 创建一个二元交叉熵损失函数实例
criterion = nn.BCELoss()

# 计算二元交叉熵损失
loss = criterion(predicted, actual)

# 打印损失值
print(loss)
```

> BCE Loss 常用于二分类问题中，用于衡量模型的预测概率与真实标签之间的差异，以便优化模型的参数以最小化损失。