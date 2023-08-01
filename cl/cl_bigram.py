import torch
import torch.nn as nn
from torch.nn import functional as F

from collections import defaultdict

### hyper params
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
split = 0.9

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

### create mappings between int and char
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(f'chars = {chars}')
# print(f'len(chars) = {len(chars)}')

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda lst : ''.join([itos[i] for i in lst])
# print(encode("hello world!"))
# print(decode(encode("hello world!")))



### prepare the train and split data
data = encode(text)
train_data = torch.tensor(data[ : int(len(data) * split)])
val_data = torch.tensor(data[int(len(data) * split) : ])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    starts = torch.randint(len(data) - block_size, (batch_size, ))
    xb = torch.stack([data[start: start + block_size] for start in starts]).to(device)
    yb = torch.stack([data[start + 1 : start + block_size + 1] for start in starts]).to(device)
    return xb, yb

@torch.no_grad()
def estimate_loss():
    m.eval()
    losses = defaultdict(lambda : 0)
    for split in ('train', 'eval'):
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[split] += loss.item()
        losses[split] /= eval_iters
    return losses['train'], losses['eval']


# xb, yb = get_batch('train')
# print(f'xb = {xb}')
# print(f'yb = {yb}')

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            last = logits[ : , -1 , : ] # B * C
            probs = F.softmax(last, dim = -1)
            preds = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, preds), dim = 1)
        return idx

m = BigramLanguageModel(vocab_size).to(device)
# logits, loss = m(xb, yb)
# print(logits.shape)

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0:
        train_loss, val_loss = estimate_loss()
        print(f'Iter {i}, train_loss = {train_loss}, val_loss = {val_loss}')
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

idx = torch.zeros([1, 1], dtype=torch.long, device=device)
res = decode(m.generate(idx, 500)[0].tolist())
print(f'res = {res}')
