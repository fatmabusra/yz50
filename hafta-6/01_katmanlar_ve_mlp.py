import torch
import torch.nn.functional as F
import random

# -------------------------------------------------------------
# 1. VERİ SETİNİ OKU VE SÖZLÜĞÜ HAZIRLAMA
# -------------------------------------------------------------
words = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
words = [w.lower().strip() for w in words if w.strip()]

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)

# Train (%80), Dev (%10), Test (%10) ayrımı
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
# 2. PYTORCH BENZERİ KENDİ KATMAN SINIFLARIMIZ (GÖREV 1)
# -------------------------------------------------------------

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming He normalizasyonu ile ağırlık ilklendirme
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
        # Öğrenilebilir parametreler (gain ve bias)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Çıkarım (inference) için hareketli istatistikler
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            # 2D veya 3D girdi ayrımı
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True, unbiased=False)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # İstatistikleri güncelledik (sadece eğitimde)
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
        # Tüm alt katmanların parametrelerini tek bir listede toplar
        return [p for layer in self.layers for p in layer.parameters()]


# -------------------------------------------------------------
# 3. MODELİ EĞİTME VE DEĞERLENDİRME FONKSİYONU
# -------------------------------------------------------------
def train_and_evaluate(block_size, model, steps=20000, batch_size=32):
    # Veri setlerini oluştur
    Xtr, Ytr = build_dataset(words[:n1], block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size)

    # Parametreleri topla ve gradyan takibini aç
    parameters = model.parameters()
    total_params = sum(p.nelement() for p in parameters)
    for p in parameters:
        p.requires_grad = True

    # Eğitim döngüsü
    g = torch.Generator().manual_seed(42)
    for step in range(steps):
        # Minibatch
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # Forward pass (Model tek bir fonksiyon gibi çağrılır)
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Güncelleme (Learning rate decay)
        lr = 0.1 if step < 15000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

    # Değerlendirme (Dev Loss)
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = False

    with torch.no_grad():
        dev_logits = model(Xdev)
        dev_loss = F.cross_entropy(dev_logits, Ydev).item()

    return total_params, dev_loss


# -------------------------------------------------------------
# 4. GÖREV 2: BAĞLAM 3 VS BAĞLAM 8 KARŞILAŞTIRMASI (BASELINE)
# -------------------------------------------------------------
if __name__ == '__main__':
    print("="*65)
    print(" GÖREV 1 & 2: KATMAN SINIFLARI VE DÜZ MLP KARŞILAŞTIRMASI")
    print("="*65)

    n_embd = 10
    n_hidden = 68

    # 1. Model: Bağlam 3 (Hafta 4'teki Düz Yapı)
    torch.manual_seed(42)
    model_b3 = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(n=3), # 3 harfi tek seferde birleştirir
        Linear(3 * n_embd, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    params_b3, dev_loss_b3 = train_and_evaluate(block_size=3, model=model_b3, steps=10000)
    print(f"Bağlam 3 Düz MLP | Parametre: {params_b3:6d} | Dev Loss: {dev_loss_b3:.4f}")

    # 2. Model: Bağlam 8 (Sadece bağlam genişletilmiş Düz Yapı)
    torch.manual_seed(42)
    model_b8 = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(n=8), # 8 harfi tek seferde birleştirir (8 * 10 = 80 girdi)
        Linear(8 * n_embd, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    params_b8, dev_loss_b8 = train_and_evaluate(block_size=8, model=model_b8, steps=10000)
    print(f"Bağlam 8 Düz MLP | Parametre: {params_b8:6d} | Dev Loss: {dev_loss_b8:.4f}")
    print("="*65)
    print(f"Fark Özeti: Parametre {params_b8 - params_b3} arttı. Dev loss {dev_loss_b3:.4f} -> {dev_loss_b8:.4f} oldu.")