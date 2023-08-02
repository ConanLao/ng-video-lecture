import torch
import torch.nn as nn
from torch.nn import functional as F

from collections import defaultdict

### hyper params
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
split = 0.9
n_emb = 32

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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias = False)
        self.query = nn.Linear(n_emb, head_size, bias = False)
        self.value = nn.Linear(n_emb, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[ : T, : T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_emb)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        # self.sa_head = Head(n_emb)
        self.sa_head = MultiHeadAttention(4, n_emb // 4)
        self.lm_head = nn.Linear(n_emb, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # B, T
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # B, T
        emb = token_emb + pos_emb
        a = self.sa_head(emb)
        logits = self.lm_head(a)
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
            cropped = idx[ : , -block_size:]
            logits, loss = self(cropped)
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

# idx = torch.zeros([1, 1], dtype=torch.long, device=device)
# res = decode(m.generate(idx, 500)[0].tolist())
# print(f'{res}')

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
