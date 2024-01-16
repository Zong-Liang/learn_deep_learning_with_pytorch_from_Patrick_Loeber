import torch

# 1d
x = torch.empty(1)
print(x)
x = torch.empty(3)
print(x)

# 2d
x = torch.empty(2, 3)
print(x)

# 3d
x = torch.empty(1, 2, 3)
print(x)

# 4d
x = torch.empty(1, 1, 2, 3)
print(x)

x = torch.rand(2, 2)
print(x)

x = torch.zeros(2, 2)
print(x)

x = torch.ones(2, 2)
print(x)
print(x.dtype)

x = torch.ones(2, 2, dtype=torch.int)
print(x)
print(x.size())

x = torch.tensor([2.5, 0.1])
print(x)

print("-" * 80)

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)

# +
z = x + y
z = torch.add(x, y)
print(z)
y.add_(x)
print(y)

# -
z = x - y
z = torch.sub(x, y)
print(z)
y.sub_(x)
print(y)

# *
z = x * y
z = torch.mul(x, y)
print(z)
y.mul_(x)
print(y)

# /
z = x / y
z = torch.div(x, y)
print(z)
y.div_(x)
print(y)

print("-" * 80)

# slice
x = torch.rand(5, 3)
print(x)
print(x[:, 0])
print(x[1, :])
print(x[1, 1])
print(x[1, 1].item())  # use item() only when there is one elemnet in tensor

print("-" * 80)

# reshape
x = torch.rand(4, 4)
print(x)
y = x.view(16)
print(y)
y = x.view(-1, 8)
print(y.size())

print("-" * 80)

# from numpy array to torch tensor and vice versa
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)  # a b share the same memory location

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)

print("-" * 80)

# use cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.zeros(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")
    z = z.numpy()
    print(z)

print("-" * 80)

# requires_grad
x = torch.rand(5, requires_grad=True)
print(x)
