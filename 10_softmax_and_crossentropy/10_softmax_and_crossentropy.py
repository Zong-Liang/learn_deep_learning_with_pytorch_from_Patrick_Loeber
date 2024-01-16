import numpy as np
import torch
from torch import nn


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print("achieve softmax by hand: ")
x = np.array([2.0, 1.0, 0.1])
print(f"np_x: {x}")
outputs = softmax(x)
print(outputs)
print("-----------------------------------------------------------")

print("use torch.softmax(): ")
x = torch.tensor([2.0, 1.0, 0.1])
print(f"tensor_x: {x}")
outputs = torch.softmax(x, dim=0)
print(outputs)
print("-----------------------------------------------------------")


def cross_entropy(actual, predicted):
    return -np.sum(actual * np.log(predicted))


print("achieve cross_entropy by hand: ")
# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
y = np.array([1, 0, 0])

# y_pred has probabilities
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)
print(f"loss_1: {l1:.4f}")
print(f"loss_2: {l2:.4f}")
print("-----------------------------------------------------------")

print("use torch.nn.CrossEntropyLoss(): ")
loss = nn.CrossEntropyLoss()

# 1 samples
y = torch.tensor([0])
# n_samples × n_classes = 1 × 3
y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)
print(f"loss_1: {l1.item():.4f}")
print(f"loss_2: {l2.item():.4f}")
_, predictions_1 = torch.max(y_pred_good, 1)
_, predictions_2 = torch.max(y_pred_bad, 1)
print(predictions_1)
print(predictions_2)


# 3 samples
y = torch.tensor([2, 0, 1])
# n_samples × n_classes = 3 × 3
y_pred_good = torch.tensor([[2.0, 1.0, 3.1], [2.0, 1.0, 0.1], [2.0, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1], [2.0, 1.0, 0.1]])
l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)
print(f"loss_1: {l1.item():.4f}")
print(f"loss_2: {l2.item():.4f}")
_, predictions_1 = torch.max(y_pred_good, 1)
_, predictions_2 = torch.max(y_pred_bad, 1)
print(predictions_1)
print(predictions_2)
print("-----------------------------------------------------------")


print("# binary classification: ")


class NN_1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN_1, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # sigmoid at the end
        return torch.sigmoid(self.linear_2(self.relu(self.linear_1(x))))


model = NN_1(input_size=28 * 28, hidden_size=5)
criterion = nn.BCELoss()


print("-----------------------------------------------------------")
print("# multi classification: ")


class NN_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN_2, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # no sigmoid at the end
        return self.linear_2(self.relu(self.linear_1(x)))


model = NN_2(input_size=28 * 28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # applies softmax
