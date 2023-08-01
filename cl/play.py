import torch

# a = torch.tensor([1,1,1])
# print(a.item())


b = torch.ones((3,3))
print(torch.sum(b, 1, keepdim=True))

c = torch.tril(torch.ones(3, 3))

c = c / torch.sum(c, dim = 1, keepdim=True)
print(c)