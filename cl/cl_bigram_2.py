import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
dev_ratio = 0.9
learning_rate = 1e-2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(vocab_size)

# tokenization
stoi = {c : i for i, c in enumerate(chars)}
itos = {i : c for i, c in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda lst : ''.join([itos[i] for i in lst])

print(encode("hello world!"))
print(decode(encode("hello world!")))

data = encode(text)
dev_len = int(dev_ratio * len(data))
train_data = data[ : dev_len]
val_data = data[dev_len : ]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    indices = torch.randint(len(data) - block_size, (batch_size, ))
    xb = torch.stack([torch.tensor(data[st : st + block_size]) for st in indices]).to(device)
    yb = torch.stack([torch.tensor(data[st + 1 : st + block_size + 1]) for st in indices]).to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss():
    m.eval()
    metrics = {'train' : 0, 'val' : 0}
    for split in ('train', 'val'):
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            metrics[split] += loss.item()
    m.train()
    return metrics['train'] / eval_iters, metrics['val'] / eval_iters

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indices, targets=None):
        logits = self.embedding(indices)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_len):
        for i in range(max_len):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim = 1)
        return idx

m = BigramModel(vocab_size).to(device)

# xb, yb = get_batch('train')
# print(xb, yb)

# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        train_loss, val_loss = estimate_loss()
        print(f'iter {i}, train_loss = {train_loss}, val_loss = {val_loss}')
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

idx = torch.zeros([1, 1], dtype=torch.long, device=device)
res = decode(m.generate(idx, 500)[0].tolist())
print(res)