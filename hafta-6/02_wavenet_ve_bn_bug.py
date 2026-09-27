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
block_size = 8 # 8 harflik bağlam

# Train (%80), Dev (%10), Test (%10)
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

def build_dataset(words_subset):
    X, Y = [], []
    for w in words_subset:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])

# -------------------------------------------------------------
# 2. KATMAN SINIFLARI
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


# Hatalı ve Düzeltilmiş BatchNorm'u kıyaslayabilmek için esnek sınıf
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1, fix_bug=True):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.fix_bug = fix_bug # Hatayı kapatıp açma anahtarı
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                # DİKKAT: Hata buradaydı!
                # fix_bug=False ise sadece 0. eksende alır (Hatalı Karpathy hali)
                # fix_bug=True ise (0, 1) eksenlerinde alır (Doğru düzeltilmiş hal)
                dim = (0, 1) if self.fix_bug else 0

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


# İkişer ikişer birleştiren kilit katmanımız
class FlattenConsecutive:
    def __init__(self, n=2):
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
# 3. GÖREV 3: WAVENET AĞACI VE KATMAN ŞEKİLLERİ
# -------------------------------------------------------------
print("="*65)
print(" GÖREV 3: WAVENET KATMAN BOYUTLARININ TAKİBİ")
print("="*65)

n_embd = 10
n_hidden = 68

# WaveNet Modeli: 8 harfi ikişer ikişer birleştirir
layers = [
    Embedding(vocab_size, n_embd),          # [32, 8] -> [32, 8, 10]
    FlattenConsecutive(2),                  # [32, 8, 10] -> [32, 4, 20]
    Linear(n_embd * 2, n_hidden, bias=False),# [32, 4, 68]
    BatchNorm1d(n_hidden, fix_bug=True),
    Tanh(),
    FlattenConsecutive(2),                  # [32, 4, 68] -> [32, 2, 136]
    Linear(n_hidden * 2, n_hidden, bias=False),# [32, 2, 68]
    BatchNorm1d(n_hidden, fix_bug=True),
    Tanh(),
    FlattenConsecutive(2),                  # [32, 2, 68] -> [32, 136] (tek vektör oldu)
    Linear(n_hidden * 2, n_hidden, bias=False),# [32, 68]
    BatchNorm1d(n_hidden, fix_bug=True),
    Tanh(),
    Linear(n_hidden, vocab_size),           # [32, vocab_size]
]

wavenet = Sequential(layers)

# Şekilleri görmek için küçük bir deneme girdisi geçirelim
X_ornek = Xtr[:32] # 32 örnek
x = X_ornek
print(f"Girdi X Şekli         : {list(x.shape)}")
for i, layer in enumerate(wavenet.layers):
    x = layer(x)
    print(f"Katman {i+1:2d} ({layer.__class__.__name__:18s}) Çıktı Şekli: {list(x.shape)}")

# -------------------------------------------------------------
# 4. GÖREV 4: BATCHNORM HATASI OLAN VE DÜZELTİLEN MODELİ EĞİTME
# -------------------------------------------------------------
print("\n" + "="*65)
print(" GÖREV 4: BATCHNORM BUG'LI VE DÜZELTİLMİŞ MODEL KIYASI")
print("="*65)

def train_model(model, steps=10000):
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True
    
    g = torch.Generator().manual_seed(42)
    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]
        
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        
        for p in parameters:
            p.grad = None
        loss.backward()
        
        lr = 0.1 if step < 7500 else 0.01
        for p in parameters:
            p.data += -lr * p.grad
            
    # Dev loss
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False
    with torch.no_grad():
        dev_loss = F.cross_entropy(model(Xdev), Ydev).item()
    return dev_loss

# 1. Hatalı BatchNorm (fix_bug=False)
torch.manual_seed(42)
wavenet_bugli = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden, fix_bug=False), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden, fix_bug=False), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden, fix_bug=False), Tanh(),
    Linear(n_hidden, vocab_size)
])
loss_bugli = train_model(wavenet_bugli, steps=10000)
print(f"BatchNorm BUG'lı WaveNet Dev Loss : {loss_bugli:.4f}")

# 2. Düzeltilmiş BatchNorm (fix_bug=True)
torch.manual_seed(42)
wavenet_duzgun = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden, fix_bug=True), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden, fix_bug=True), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden, fix_bug=True), Tanh(),
    Linear(n_hidden, vocab_size)
])
loss_duzgun = train_model(wavenet_duzgun, steps=10000)
print(f"Düzeltilmiş BatchNorm WaveNet Dev Loss : {loss_duzgun:.4f}")
print("="*65)