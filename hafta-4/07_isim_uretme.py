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

# 2. X ve Y veri setini hazırla
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X, Y = torch.tensor(X), torch.tensor(Y)

# 3. Model Parametreleri (Kaiming Init + BatchNorm)
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g) * ((5/3) / ((block_size * emb_dim)**0.5))
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

# 4. Modeli Eğit
print("Model eğitiliyor, lütfen birkaç saniye bekleyin...")
for i in range(1500):
    ix = torch.randint(0, X.shape[0], (32,), generator=g)
    emb = C[X[ix]].view(-1, block_size * emb_dim)
    hpreact = emb @ W1
    
    bnmean = hpreact.mean(0, keepdim=True)
    bnstd = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmean) / (bnstd + 1e-5) + bnbias
    
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstd
        
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    
    for p in parameters:
        p.grad = None
    loss.backward()
    
    lr = 0.1 if i < 1000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

print(f"Eğitim bitti! Son Loss: {loss.item():.4f}\n")

# 5. Modelden Yeni Türkçe İsimler Üret (Sampling)
print("==================================================")
print("  BU HAFTAKİ MLP MODELİNİN ÜRETTİĞİ İSİMLER")
print("==================================================")
for _ in range(7):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])].view(1, -1)
        hpreact = emb @ W1
        hpreact = bngain * (hpreact - bnmean_running) / (bnstd_running + 1e-5) + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    print("- " + ''.join(out))

print("\n==================================================")
print("  GEÇEN HAFTAKİ BİGRAM MODELİYLE KARŞILAŞTIRMA")
print("==================================================")
print("Bigram: Sadece bir önceki harfe baktığı için genelde anlamsız sesler çıkarır (örn: 'e', 'a', 'mtt').")
print("MLP (3-harf bağlamlı): Türkçe hece yapısını (ünlü-ünsüz uyumu) çok daha iyi korur ve telaffuz edilebilir isimler türetir.")