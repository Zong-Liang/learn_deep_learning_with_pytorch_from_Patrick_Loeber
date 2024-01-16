import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}.")

# set hyperparameters
input_size = 784  # 28×28
hidden_size = 64
num_classes = 10  # 0~9
num_epochs = 2
batch_size = 32
learning_rate = 0.001

# MNIST
train_data = torchvision.datasets.MNIST(
    root="../data", train=True, transform=transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(
    root="../data", train=False, transform=transforms.ToTensor(), download=False
)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)
# print(samples[0][0], labels[0])

# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(samples[i][0], cmap="gray")
# plt.show()


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.l_1 = nn.Linear(input_size, hidden_size)
        self.l_2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.l_2(torch.relu(self.l_1(x)))


# create model
model = NN(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        # 32, 1, 28, 28
        # 32, 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        # calculate loss
        loss = criterion(outputs, labels)
        # zero grad
        optimizer.zero_grad()
        # loss backward
        loss.backward()
        # optimizer step
        optimizer.step()

        if (i + 1) % 32 == 0:
            print(
                f"epoch [{epoch+1}/{num_epochs}] | step [{i+1}/{n_total_steps}] | loss: {loss.item():.4f}"
            )

# testing loop
with torch.no_grad():
    model.eval()
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"test_accuracy: {acc}")
