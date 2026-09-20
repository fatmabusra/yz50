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

# 3. Model Parametreleri (requires_grad tamamen kapalı)
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g) * ((5/3) / ((block_size * emb_dim)**0.5))
b1 = torch.randn((1, n_hidden), generator=g) * 0.1
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn((1, vocab_size), generator=g) * 0.1

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = False

batch_size = 32

print("="*60)
print(" SIFIR AUTOGRAD (LOSS.BACKWARD OLMADAN) MODEL EĞİTİMİ")
print("="*60)

# 4. Eğitim Döngüsü (1000 Adım)
for step in range(1000):
    
    # Minibatch
    ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
    Xb, Yb = X[ix], Y[ix]
    
    # Forward Pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    
    hprebn = embcat @ W1 + b1
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=False)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    h = torch.tanh(hpreact)
    
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)
    
    # Manuel Backward Pass (Sıfır Autograd)
    # a) Hızlı Cross-Entropy
    dlogits = F.softmax(logits, dim=1)
    dlogits[range(batch_size), Yb] -= 1.0
    dlogits /= batch_size
    
    # b) Çıkış Katmanı
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0, keepdim=True)
    dh = dlogits @ W2.T
    
    # c) Tanh
    dhpreact = (1.0 - h**2) * dh
    
    # d) Hızlı BatchNorm
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnbias = dhpreact.sum(0, keepdim=True)
    
    dhprebn = (bngain * bnvar_inv / batch_size) * (
        batch_size * dhpreact - dhpreact.sum(0, keepdim=True) - bnraw * (dhpreact * bnraw).sum(0, keepdim=True)
    )
    
    # e) Giriş Katmanı
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0, keepdim=True)
    dembcat = dhprebn @ W1.T
    demb = dembcat.view(emb.shape)
    
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            dC[Xb[k, j]] += demb[k, j]
            
    # SGD Parametre Güncellemesi
    grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
    lr = 0.1 if step < 700 else 0.01
    
    for p, grad in zip(parameters, grads):
        p.data -= lr * grad
        
    if step % 200 == 0 or step == 999:
        print(f"Adım {step:4d} | Loss: {loss.item():.4f}")

print("\nModel loss.backward() olmadan, tamamen kendi türevlerimizle başarıyla eğitildi!")