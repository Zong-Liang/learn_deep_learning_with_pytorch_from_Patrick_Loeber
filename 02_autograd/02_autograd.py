import torch

torch.manual_seed(42)

x = torch.randn(3, requires_grad=True)
print(x)  # tensor([0.3367, 0.1288, 0.2345], requires_grad=True)

y = x + 2
print(y)  # tensor([2.3367, 2.1288, 2.2345], grad_fn=<AddBackward0>)

z = y * y * 2
print(z)
z = z.mean()  # tensor([10.9202,  9.0637,  9.9856], grad_fn=<MulBackward0>)
print(z)  # tensor(9.9898, grad_fn=<MeanBackward0>)

z.backward()  # dz/dx
print(x.grad)  # tensor([3.1156, 2.8384, 2.9793])


# x.requires_grad_(False)
x.requires_grad_(False)
print(x)  # tensor([0.3367, 0.1288, 0.2345])

# x.detach()
x = x.detach()
print(x)  # tensor([0.3367, 0.1288, 0.2345])

# with torch.no_grad():
with torch.no_grad():
    y = x + 2
    print(y)  # tensor([2.3367, 2.1288, 2.2345])

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)
    # tensor([3., 3., 3., 3.])
    # tensor([6., 6., 6., 6.])
    # tensor([9., 9., 9., 9.])

    weights.grad.zero_()
