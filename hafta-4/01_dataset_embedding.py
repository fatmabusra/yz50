import torch

# 1. Veri setini oku
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

# Karakter haritası
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

# 2. Bağlam Penceresi (block_size = 3)
block_size = 3
X, Y = [], []

for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

# 3. 2 Boyutlu Embedding Tablosu 
g = torch.Generator().manual_seed(42)
C = torch.randn((vocab_size, 2), generator=g)

print("X boyutu:", X.shape)
print("Y boyutu:", Y.shape)
print("İlk 3 harfin embedding'i:\n", C[X[0]])