import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_emb = 32
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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, xb_emb):
        B, T, C = xb_emb.shape
        k = self.key(xb_emb) # B, T, head_size
        q = self.query(xb_emb) # B, T, head_size
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # B, T, T
        # dont forget [ : T, : T]; Weight matrix size should be same as the actual block size, which may vary during generation
        wei = wei.masked_fill(self.tril[ : T, : T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        v = self.value(xb_emb) # B, T, head_size
        out = wei @ v # T, T @ B, T, head_size -> B, T, head_size
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        # can't use a normal List
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_emb)

    def forward(self, xb):
        out = torch.cat([head(xb) for head in self.heads], dim = -1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, n_emb * 4),
            nn.ReLU(),
            nn.Linear(n_emb * 4, n_emb)
        )
    
    def forward(self, xb):
        return self.net(xb)
    
class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_head, n_emb // n_head)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, xb):
        xb = xb + self.sa_heads(self.ln1(xb))
        xb = xb + self.ffwd(self.ln2(xb))
        return xb

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_emb)
        self.pos_emb_table = nn.Embedding(block_size, n_emb)
        # self.sa_head = Head(n_emb)
        # self.sa_heads = MultiHeadAttention(4, n_emb // 4)
        # self.ffwd = FeedForward(n_emb)
        self.blocks = nn.Sequential(
            Block(n_emb, n_head=4),
            Block(n_emb, n_head=4),
            Block(n_emb, n_head=4),
            nn.LayerNorm(n_emb)
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, xb, yb = None):
        B, T = xb.shape
        tok_emb = self.token_emb_table(xb)
        pos_emb = self.pos_emb_table(torch.arange(T))
        emb = tok_emb + pos_emb
        a = self.blocks(emb)
        logits = self.lm_head(a)
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
            B, T = xb.shape
            logits, _ = self(xb[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim = -1)
            preds = torch.multinomial(probs, num_samples = 1)
            xb = torch.cat((xb, preds), dim = 1)
        return xb
    

m = BigramLanguageModel().to(device)

# xb, yb = get_batch('train')
# print(f'xb = {xb}')
# print(f'yb = {yb}')
# m(xb)

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
for i in range(max_iters):
    if i % eval_interval == 0 or i == max_iters - 1:
        # estimate_loss will affect random because it calls get_batch which randomly creates batch
        train_loss, val_loss = estimate_loss(m)
        print(f'iter {i}: train_loss = {train_loss}, val_loss = {val_loss}')
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


res = decode(m.generate(torch.zeros((1, 1), dtype =torch.long, device=device), 500)[0].tolist())
print(res)