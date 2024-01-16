import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameters
num_epochs = 2
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]
# we transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = train_dataset.classes


# design model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv_1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv_2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc_1(x))  # -> n, 120
        x = F.relu(self.fc_2(x))  # -> n, 84
        x = self.fc_3(x)  # -> n, 10
        return x


model = CNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original shape [4, 3, 32, 32] = [4, 3, 1024]
        # input_layer 3 input channels, 6 output channels, 5 kernel size
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(
                f"epoch [{epoch+1}/{num_epochs}] | step [{i+1}/{n_total_steps}] | loss: {loss.item():.4f}"
            )
print("training finished!")
print("----------------------------------------------------------")

# 定义了保存模型的文件夹路径
MODEL_PATH = Path("../checkpoints")
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

print("----------------------------------------------------------")

with torch.inference_mode():
    model.eval()
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f"test_accuracy: {acc}%")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"test_accuracy_of_{classes[i]}: {acc}%")
