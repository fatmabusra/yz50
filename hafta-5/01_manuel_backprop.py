import torch
import torch.nn.functional as F

# -------------------------------------------------------------
# 1. VERİ SETİNİ OKU VE SÖZLÜĞÜ HAZIRLA
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 2. X VE Y VERİ SETİNİ HAZIRLA
# -------------------------------------------------------------
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X, Y = torch.tensor(X), torch.tensor(Y)

# Karpathy'nin Karşılaştırma Fonksiyonu (cmp)
def cmp(s, dt, t):
    exact = torch.all(dt == t.grad).item()
    approx = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:18s} | exact: {str(exact):5s} | approx: {str(approx):5s} | maxdiff: {maxdiff}')

# -------------------------------------------------------------
# 3. MODEL PARAMETRELERİ
# -------------------------------------------------------------
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g) * ((5/3) / ((block_size * emb_dim)**0.5))
b1 = torch.randn(n_hidden, generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

# Hatanın çözümü burası: Parametrelerin gradyan takibini açıyoruz
parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

# Küçük bir batch (32 örnek)
batch_size = 32
ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
Xb, Yb = X[ix], Y[ix]

# -------------------------------------------------------------
# 4. İLERİ YAYILIM (FORWARD PASS) - ATOMİK ADIMLAR
# -------------------------------------------------------------
# Embedding
emb = C[Xb]                                       # [32, 3, 2]
embcat = emb.view(emb.shape[0], -1)               # [32, 6]

# Linear + BatchNorm + Tanh
hprebn = embcat @ W1 + b1                         # [32, 100]
bnmeani = 1.0 / batch_size * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1.0 / (batch_size - 1) * (bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
h = torch.tanh(hpreact)                           # [32, 100]

# Logits
logits = h @ W2 + b2                              # [32, vocab_size]

# Cross-Entropy'nin atomik adımları
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss_all = logprobs[range(batch_size), Yb]
loss = -loss_all.mean()

# PyTorch autograd karşılaştırması için değişkenlerin gradyanlarını tut
for t in [loss_all, logprobs, probs, counts_sum_inv, counts_sum, counts, 
          norm_logits, logit_maxes, logits, h, hpreact, bnraw, bnvar_inv, 
          bnvar, bndiff2, bndiff, bnmeani, hprebn, embcat, emb]:
    t.retain_grad()

loss.backward()

# -------------------------------------------------------------
# 5. GERİYE YAYILIM (MANUEL BACKPROP) VE CMP İLE DOĞRULAMA
# -------------------------------------------------------------
print("="*65)
print(" MANUEL TÜREVLER VE PYTORCH AUTOGRAD KARŞILAŞTIRMASI (CMP)")
print("="*65)

# 1. loss -> loss_all
dloss_all = (-1.0 / batch_size) * torch.ones_like(loss_all)
cmp('loss_all', dloss_all, loss_all)

# 2. loss_all -> logprobs
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(batch_size), Yb] = dloss_all
cmp('logprobs', dlogprobs, logprobs)

# 3. logprobs -> probs (d/dx log(x) = 1/x)
dprobs = (1.0 / probs) * dlogprobs
cmp('probs', dprobs, probs)

# 4. probs -> counts_sum_inv ve counts (Çarpım türevi + broadcasting sum)
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

dcounts = counts_sum_inv * dprobs

# 5. counts_sum_inv -> counts_sum (d/dx x^-1 = -x^-2)
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)

# 6. counts_sum -> counts (counts iki farklı dala gitmişti, türevler toplanır)
dcounts += torch.ones_like(counts) * dcounts_sum
cmp('counts', dcounts, counts)

# 7. counts -> norm_logits (d/dx exp(x) = exp(x))
dnorm_logits = counts * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)

# 8. norm_logits -> logit_maxes
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes)

# 9. norm_logits + logit_maxes -> logits
dlogits = dnorm_logits.clone()
dlogits += F.one_hot(logits.max(1).indices, num_classes=vocab_size) * dlogit_maxes
cmp('logits', dlogits, logits)

# 10. logits -> h, W2, b2
dh = dlogits @ W2.T
cmp('h', dh, h)
dW2 = h.T @ dlogits
cmp('W2', dW2, W2)
db2 = dlogits.sum(0, keepdim=True)
cmp('b2', db2, b2)

# 11. h -> hpreact (d/dx tanh(x) = 1 - tanh^2(x))
dhpreact = (1.0 - h**2) * dh
cmp('hpreact', dhpreact, hpreact)

# 12. BatchNorm Geriye Yayılımı
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
cmp('bngain', dbngain, bngain)
dbnbias = dhpreact.sum(0, keepdim=True)
cmp('bnbias', dbnbias, bnbias)

dbnraw = bngain * dhpreact
cmp('bnraw', dbnraw, bnraw)

dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)

dbndiff2 = (1.0 / (batch_size - 1)) * torch.ones_like(bndiff2) * dbnvar
cmp('bndiff2', dbndiff2, bndiff2)

dbndiff = bnvar_inv * dbnraw + (2.0 * bndiff) * dbndiff2
cmp('bndiff', dbndiff, bndiff)

dbnmeani = (-dbndiff).sum(0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

dhprebn = dbndiff.clone() + (1.0 / batch_size) * torch.ones_like(hprebn) * dbnmeani
cmp('hprebn', dhprebn, hprebn)

# 13. Giriş Katmanı Parametreleri (W1, b1, embcat, C)
dW1 = embcat.T @ dhprebn
cmp('W1', dW1, W1)
db1 = dhprebn.sum(0, keepdim=True)
cmp('b1', db1, b1)

dembcat = dhprebn @ W1.T
cmp('embcat', dembcat, embcat)

demb = dembcat.view(emb.shape)
cmp('emb', demb, emb)

dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix_ch = Xb[k, j]
        dC[ix_ch] += demb[k, j]
cmp('C', dC, C)