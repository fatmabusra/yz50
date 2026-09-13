import torch
import torch.nn.functional as F
import random

# 1. Veri setini oku ve Train / Dev olarak böl
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
vocab_size = len(stoi)
block_size = 3
emb_dim = 2
n_hidden = 100

def build_dataset(words_list):
    X_out, Y_out = [], []
    for w in words_list:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X_out.append(context)
            Y_out.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X_out), torch.tensor(Y_out)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:])

# ==========================================
# MODEL 1: BatchNorm OLMADAN EĞİTİM
# ==========================================
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g) * ((5/3) / ((block_size * emb_dim)**0.5))
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0
params_no_bn = [C, W1, b1, W2, b2]
for p in params_no_bn: p.requires_grad = True

for i in range(1200):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    emb = C[Xtr[ix]].view(-1, block_size * emb_dim)
    h = torch.tanh(emb @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    for p in params_no_bn: p.grad = None
    loss.backward()
    lr = 0.1 if i < 800 else 0.01
    for p in params_no_bn: p.data += -lr * p.grad

emb_dev = C[Xdev].view(-1, block_size * emb_dim)
h_dev = torch.tanh(emb_dev @ W1 + b1)
loss_no_bn = F.cross_entropy(h_dev @ W2 + b2, Ydev).item()

# ==========================================
# MODEL 2: BatchNorm İLE EĞİTİM
# ==========================================
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g) * ((5/3) / ((block_size * emb_dim)**0.5))
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

# BatchNorm Parametreleri (b1 gerekmez çünkü bnbias onun yerini alır)
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

params_bn = [C, W1, W2, b2, bngain, bnbias]
for p in params_bn: p.requires_grad = True

for i in range(1200):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    emb = C[Xtr[ix]].view(-1, block_size * emb_dim)
    hpreact = emb @ W1
    
    # BatchNorm Adımı
    bnmean = hpreact.mean(0, keepdim=True)
    bnstd = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmean) / (bnstd + 1e-5) + bnbias
    
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstd
        
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    
    for p in params_bn: p.grad = None
    loss.backward()
    lr = 0.1 if i < 800 else 0.01
    for p in params_bn: p.data += -lr * p.grad

# Dev Loss (Running mean ve std kullanılarak)
emb_dev = C[Xdev].view(-1, block_size * emb_dim)
hpreact_dev = emb_dev @ W1
hpreact_dev = bngain * (hpreact_dev - bnmean_running) / (bnstd_running + 1e-5) + bnbias
h_dev = torch.tanh(hpreact_dev)
loss_bn = F.cross_entropy(h_dev @ W2 + b2, Ydev).item()

print("--- KARŞILAŞTIRMA SONUÇLARI ---")
print(f"BatchNorm'suz Dev Loss: {loss_no_bn:.4f}")
print(f"BatchNorm'lu  Dev Loss: {loss_bn:.4f}")