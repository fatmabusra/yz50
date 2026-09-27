import torch
import torch.nn.functional as F
import random

# -------------------------------------------------------------
# 1. VERİ SETİ HAZIRLIĞI
# -------------------------------------------------------------
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

# %80 Eğitim, %10 Doğrulama (Dev), %10 Test
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

def build_dataset(words_subset, block_size):
    X, Y = [], []
    for w in words_subset:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

# -------------------------------------------------------------
# 2. KATMAN SINIFLARI (Düzeltilmiş BatchNorm Dahil)
# -------------------------------------------------------------
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / (fan_in**0.5)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim == 2 else (0, 1) # Düzeltilmiş BatchNorm eksenleri
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=False)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.squeeze()
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    def parameters(self):
        return [self.weight]

class FlattenConsecutive:
    def __init__(self, n=1):
        self.n = n
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# -------------------------------------------------------------
# 3. GENEL EĞİTİM VE DEĞERLENDİRME FONKSİYONU
# -------------------------------------------------------------
def train_and_eval(model, block_size, steps=10000, batch_size=32):
    Xtr, Ytr = build_dataset(words[:n1], block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size)

    parameters = model.parameters()
    total_params = sum(p.nelement() for p in parameters)
    for p in parameters:
        p.requires_grad = True

    g = torch.Generator().manual_seed(42)
    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None
        loss.backward()

        lr = 0.1 if step < int(0.75 * steps) else 0.01
        for p in parameters:
            p.data += -lr * p.grad

    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False

    with torch.no_grad():
        dev_loss = F.cross_entropy(model(Xdev), Ydev).item()

    return total_params, dev_loss

# -------------------------------------------------------------
# 4. GÖREV 5: KARŞILAŞTIRMA TABLOSU OLUŞTURMA
# -------------------------------------------------------------
print("="*65)
print(" GÖREV 5: 3 MODELİN KARŞILAŞTIRMA TABLOSU HAZIRLANIYOR...")
print("="*65)

# Model 1: Bağlam 3 Düz MLP
torch.manual_seed(42)
m1 = Sequential([
    Embedding(vocab_size, 10),
    FlattenConsecutive(3),
    Linear(3 * 10, 68, bias=False), BatchNorm1d(68), Tanh(),
    Linear(68, vocab_size)
])
p1, l1 = train_and_eval(m1, block_size=3, steps=10000)

# Model 2: Bağlam 8 Düz MLP
torch.manual_seed(42)
m2 = Sequential([
    Embedding(vocab_size, 10),
    FlattenConsecutive(8),
    Linear(8 * 10, 68, bias=False), BatchNorm1d(68), Tanh(),
    Linear(68, vocab_size)
])
p2, l2 = train_and_eval(m2, block_size=8, steps=10000)

# Model 3: Bağlam 8 WaveNet (Genişletilmiş / Ölçeklenmiş)
n_embd = 24
n_hidden = 128
torch.manual_seed(42)
wavenet_big = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
])
p3, l3 = train_and_eval(wavenet_big, block_size=8, steps=10000)

# 3 Satırlık Karşılaştırma Tablosunu Ekrana Yazdır
print("\n" + "="*50)
print(f"{'Model Mimarisi':<25} | {'Parametre':<10} | {'Dev Loss':<8}")
print("-" * 50)
print(f"{'Bağlam 3 Düz MLP':<25} | {p1:<10d} | {l1:<8.4f}")
print(f"{'Bağlam 8 Düz MLP':<25} | {p2:<10d} | {l2:<8.4f}")
print(f"{'Bağlam 8 WaveNet (Büyük)':<25} | {p3:<10d} | {l3:<8.4f}")
print("="*50)

# -------------------------------------------------------------
# 5. GÖREV 6: TÜRKÇE İSİM ÜRETİMİ (SAMPLING)
# -------------------------------------------------------------
print("\n" + "="*65)
print(" GÖREV 6: WAVENET İLE TÜRETİLEN TÜRKÇE İSİMLER")
print("="*65)

g = torch.Generator().manual_seed(42)
for _ in range(10):
    out = []
    context = [0] * 8 # 8 harflik boş bağlamla başla
    while True:
        x = torch.tensor([context])
        logits = wavenet_big(x)
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            break
        out.append(itos[ix])
    print(''.join(out))