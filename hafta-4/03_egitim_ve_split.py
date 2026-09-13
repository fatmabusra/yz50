import torch
import torch.nn.functional as F
import random

# 1. Veri setini oku
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
vocab_size = len(stoi)
block_size = 3

# 2. Veri seti oluşturma fonksiyonu
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

# 3. Train / Dev / Test Ayrımı (%80, %10, %10)
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

print(f"Eğitim seti boyutu: {Xtr.shape[0]}")
print(f"Dev (Validation) seti boyutu: {Xdev.shape[0]}")

# 4. Model Parametreleri
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

# 5. Minibatch ile Eğitim (1000 Adım)
for i in range(1000):
    # Minibatch (32 örnek)
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # Forward pass
    emb = C[Xb].view(-1, block_size * emb_dim)
    h = torch.tanh(emb @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Learning rate decay: başta 0.1, son 300 adımda 0.01
    lr = 0.1 if i < 700 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

# 6. Dev Loss Hesaplama
emb_dev = C[Xdev].view(-1, block_size * emb_dim)
h_dev = torch.tanh(emb_dev @ W1 + b1)
logits_dev = h_dev @ W2 + b2
dev_loss = F.cross_entropy(logits_dev, Ydev)

print(f"Son Minibatch Loss: {loss.item():.4f}")
print(f"Dev (Validation) Loss: {dev_loss.item():.4f}")