import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Veri setini oku
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
vocab_size = len(stoi)
block_size = 3
emb_dim = 2
n_hidden = 100

X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X, Y = torch.tensor(X), torch.tensor(Y)

# --- 1. DURUM: DÜZELTME OLMADAN (Kötü Başlatma) ---
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, emb_dim), generator=g)
W1_bad = torch.randn((block_size * emb_dim, n_hidden), generator=g)
b1_bad = torch.randn(n_hidden, generator=g)
W2_bad = torch.randn((n_hidden, vocab_size), generator=g)
b2_bad = torch.randn(vocab_size, generator=g)

emb_sample = C[X[:500]].view(-1, block_size * emb_dim)
h_bad = torch.tanh(emb_sample @ W1_bad + b1_bad)
logits_bad = h_bad @ W2_bad + b2_bad
loss_bad = F.cross_entropy(logits_bad, Y[:500])

saturated_bad = (h_bad.abs() > 0.99).float().mean().item()
print(f"[KÖTÜ INIT] Başlangıç Loss: {loss_bad.item():.4f}")
print(f"[KÖTÜ INIT] Tanh Doygunluk Oranı (Ölü Nöron Riski): %{saturated_bad * 100:.2f}")

# --- 2. DURUM: KAIMING INIT & SIFIRA YAKIN ÇIKIŞ ---
fan_in = block_size * emb_dim
# Kaiming He kuralı: gain / sqrt(fan_in), tanh için gain = 5/3
W1_good = torch.randn((fan_in, n_hidden), generator=g) * ((5/3) / (fan_in**0.5))
b1_good = torch.randn(n_hidden, generator=g) * 0.01

# Çıkış katmanını neredeyse sıfır yaparak uniform dağılım elde ediyoruz
W2_good = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2_good = torch.randn(vocab_size, generator=g) * 0

h_good = torch.tanh(emb_sample @ W1_good + b1_good)
logits_good = h_good @ W2_good + b2_good
loss_good = F.cross_entropy(logits_good, Y[:500])

saturated_good = (h_good.abs() > 0.99).float().mean().item()
print(f"[KAIMING INIT] Başlangıç Loss: {loss_good.item():.4f}")
print(f"[KAIMING INIT] Tanh Doygunluk Oranı: %{saturated_good * 100:.2f}")

# Histogram Çizimi: İki durumu yan yana karşılaştır
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(h_bad.view(-1).tolist(), 50, color='crimson')
plt.title(f"Kötü Init: Tanh Doygunluğu\n(-1 ve 1'e yığılma: %{saturated_bad*100:.1f})")
plt.xlabel("Aktivasyon Değeri")
plt.ylabel("Nöron Sayısı")

plt.subplot(1, 2, 2)
plt.hist(h_good.view(-1).tolist(), 50, color='forestgreen')
plt.title(f"Kaiming Init: Düzgün Dağılım\n(-1 ve 1'e yığılma: %{saturated_good*100:.1f})")
plt.xlabel("Aktivasyon Değeri")

plt.tight_layout()
plt.savefig("kaiming_karsilastirma.png")
print("Grafik 'kaiming_karsilastirma.png' olarak kaydedildi!")
plt.show()