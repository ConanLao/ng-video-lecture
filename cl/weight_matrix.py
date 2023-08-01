import torch
B,T,C = 4, 8, 2
A = torch.rand([B, T, C])

wei_1 = torch.tril(torch.ones((T, T)))
wei_1 = wei_1 / torch.sum(wei_1, dim = 1, keepdim = True)

print(wei_1)

tril = torch.tril(torch.ones((T, T)))
wei_2 = torch.zeros((T, T))
wei_2 = wei_2.masked_fill(tril == 0, float('-inf'))
wei_2 = torch.nn.functional.softmax(wei_2, dim = 1)
print(wei_2)

# The same as the one above, but more space efficient.
wei_3 = torch.tril(torch.ones((T, T)))
wei_3 = wei_3.masked_fill(wei_3 == 0, float('-inf'))
wei_3 = wei_3.masked_fill(wei_3 == 1, 0)
wei_3 = torch.nn.functional.softmax(wei_3, dim = 1)
print(wei_3)

# This is T, T @ B, T, C and pytorch will auto boardcast it to B, T, T @ B, T, C
# Think about broadcasting a scalar in a multiplication. We are bascally adding dimimensions to the scaler in the front.
C = wei_1 @ A 
print(C)