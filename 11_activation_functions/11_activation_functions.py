# most popular activation functions
#     step function
#     sigmoid
#     tanh
#     relu
#     leaky relu
#     softmax
import torch
from torch import nn
import torch.nn.functional as F

# F.relu()
# F.softmax()
# F.sigmoid()
# F.cross_entropy()
# F.tanh()


class NN_1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN_1, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # sigmoid at the end
        return torch.sigmoid(self.linear_2(torch.relu(self.linear_1(x))))


model = NN_1(input_size=28 * 28, hidden_size=5)
criterion = nn.BCELoss()
