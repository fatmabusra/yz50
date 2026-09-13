import torch
import torch.nn.functional as F

words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
vocab_size = len(stoi)

block_size = 3
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]
X, Y = torch.tensor(X), torch.tensor(Y)

g = torch.Generator().manual_seed(42)
emb_dim = 2
n_hidden = 100

C = torch.randn((vocab_size, emb_dim), generator=g)
W1 = torch.randn((block_size * emb_dim, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((n_hidden, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)

# Forward pass
emb = C[X].view(-1, block_size * emb_dim)
h = torch.tanh(emb @ W1 + b1)
logits = h @ W2 + b2

# Manuel Softmax & Cross Entropy
counts = logits.exp()
probs = counts / counts.sum(1, keepdims=True)
loss_manual = -probs[torch.arange(len(Y)), Y].log().mean()

# PyTorch yerleşik fonksiyon
loss_builtin = F.cross_entropy(logits, Y)

print("Manuel Loss:      ", loss_manual.item())
print("Cross Entropy Loss:", loss_builtin.item())