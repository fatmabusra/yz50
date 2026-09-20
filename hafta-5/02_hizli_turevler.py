import torch
import torch.nn.functional as F

# 1. Veri setini oku
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)
block_size = 3
emb_dim = 2
n_hidden = 100

# 2. X ve Y'yi hazırla
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X, Y = torch.tensor(X), torch.tensor(Y)

# Karşılaştırma fonksiyonu (cmp)
def cmp(s, dt, t):
    exact = torch.all(dt == t.grad).item()
    approx = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:22s} | exact: {str(exact):5s} | approx: {str(approx):5s} | maxdiff: {maxdiff}')

# 3. Model Parametreleri
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g) * ((5/3) / ((block_size * emb_dim)**0.5))
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

# 32'lik batch
batch_size = 32
ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
Xb, Yb = X[ix], Y[ix]

# 4. İleri Yayılım (Forward Pass)
emb = C[Xb]
embcat = emb.view(emb.shape[0], -1)

# Linear + BatchNorm
hprebn = embcat @ W1 + b1
bnmean = hprebn.mean(0, keepdim=True)
bnvar = hprebn.var(0, keepdim=True, unbiased=False)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = (hprebn - bnmean) * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact)

# Logits ve Loss
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yb)

# PyTorch gradyanlarını tut
for t in [logits, hpreact, hprebn]:
    t.retain_grad()
loss.backward()

print("="*65)
print(" HIZLI VE ANALİTİK TÜREVLERİN DOĞRULANMASI")
print("="*65)

# 1. Hızlı Cross-Entropy Türevi (dlogits)
dlogits = F.softmax(logits, dim=1)
dlogits[range(batch_size), Yb] -= 1.0
dlogits /= batch_size
cmp('logits (Hızlı CE)', dlogits, logits)

# 2. Hızlı BatchNorm Türevi (dhprebn)
dh = dlogits @ W2.T
dhpreact = (1.0 - h**2) * dh

dhprebn = (bngain * bnvar_inv / batch_size) * (
    batch_size * dhpreact - dhpreact.sum(0, keepdim=True) - bnraw * (dhpreact * bnraw).sum(0, keepdim=True)
)
cmp('hprebn (Hızlı BN)', dhprebn, hprebn)