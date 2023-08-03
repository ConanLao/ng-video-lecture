import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(chars)
print(vocab_size)

itos = {i: c for i, c in enumerate(chars)}
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda lst : ''.join([itos[i] for i in lst])

print(encode('hello world!'))
print(decode(encode('hello world!')))

data = encode(text)
train_data = data[ : int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)) : ]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    starts = torch.randint(len(data) - block_size, (batch_size, ))
    xb = torch.stack([torch.tensor(data[st : st + block_size]) for st in starts], dim = 0).to(device)
    yb = torch.stack([torch.tensor(data[st + 1 : st + 1 + block_size]) for st in starts], dim = 0).to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss(m):
    m.eval()
    res = {'train' :  0.0, 'val' : 0.0}
    for split in ('train', 'val'):
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = m(xb, yb)
            res[split] += loss
        res[split] /= eval_iters
    m.train()
    return res['train'], res['val']

# xb, yb = get_batch('train')
# print(f'xb = {xb}')
# print(f'yb = {yb}')

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, xb, yb = None):
        logits = self.token_emb_table(xb)
        if yb == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            yb = yb.view(B * T)
            loss = F.cross_entropy(logits, yb)
        return logits, loss

    def generate(self, xb, max_token_num):
        for _ in range(max_token_num):
            logits, _ = self(xb)
            probs = F.softmax(logits[:, -1, :], dim = -1)
            preds = torch.multinomial(probs, num_samples = 1)
            xb = torch.cat((xb, preds), dim = 1)
        return xb
    

m = BigramLanguageModel(vocab_size).to(device)

# xb, yb = get_batch('train')
# print(f'xb = {xb}')
# print(f'yb = {yb}')
# m(xb)

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
for i in range(max_iters):
    if i % eval_interval == 0:
        # estimate_loss will affect random because it calls get_batch which randomly creates batch
        train_loss, val_loss = estimate_loss(m)
        print(f'iter {i}: train_loss = {train_loss}, val_loss = {val_loss}')
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


res = decode(m.generate(torch.zeros((1, 1), dtype =torch.long, device=device), max_token_num = 500)[0].tolist())
print(res)