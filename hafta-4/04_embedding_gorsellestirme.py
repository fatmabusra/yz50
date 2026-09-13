import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Veri setini oku
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)
block_size = 3

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

# 3. Model Parametreleri (2 Boyutlu Embedding)
g = torch.Generator().manual_seed(42)
emb_dim = 2
n_hidden = 100

C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# 4. Modeli Eğit (C tablosunun harfleri öğrenmesi için)
print("Embedding uzayı eğitiliyor...")
for i in range(1500):
    ix = torch.randint(0, X.shape[0], (32,), generator=g)
    Xb, Yb = X[ix], Y[ix]

    emb = C[Xb].view(-1, block_size * emb_dim)
    h = torch.tanh(emb @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 1000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

print("Eğitim bitti, grafik çizdiriliyor...")

# 5. Harf Embedding'lerini 2 Boyutta Çizdir ve Kaydet
plt.figure(figsize=(9, 9))
plt.scatter(C[:, 0].data, C[:, 1].data, s=300, color='royalblue', alpha=0.7)

for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color="white", fontsize=10, weight='bold')

plt.title("Eğitilmiş Harf Embedding Haritası (2D)", fontsize=14)
plt.xlabel("Boyut 1")
plt.ylabel("Boyut 2")
plt.grid(True, linestyle='--', alpha=0.6)

# Resmi hem kaydet hem ekranda göster
plt.savefig("embedding_haritasi.png", bbox_inches='tight')
print("Grafik 'embedding_haritasi.png' olarak kaydedildi!")
plt.show()