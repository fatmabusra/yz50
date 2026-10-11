import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------
# 1. VERİ HAZIRLIĞI VE HİPERPARAMETRELER
# -------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Küçük ve hızlı eğitim ayarları (CPU'da hızlıca bitsin diye)
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 32

torch.manual_seed(1337)

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -------------------------------------------------------------
# 2. TEMEL YAPI TAŞLARI (ÖĞRENCİ SEVİYESİNDE YALIN KODLAR)
# -------------------------------------------------------------

# Tek bir Self-Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

# Görev 1: Multi-Head Attention (Birden çok head'i yan yana çalıştırma)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # num_heads tane bağımsız küçük head açıyoruz
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        # Her head'in çıktısını kanal boyutunda (son boyutta) yan yana birleştiriyoruz
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# Görev 2: FeedForward Katmanı (Token'ların kendi başına düşündüğü yer)
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

# Görev 2: Tam Bir Transformer Bloğu
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual bağlantı (x + ...) ve LayerNorm
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# -------------------------------------------------------------
# 3. GENEL MODEL SINIFI
# -------------------------------------------------------------
class GPTMini(nn.Module):
    def __init__(self, mimari_tipi="tam_blok"):
        super().__init__()
        self.mimari_tipi = mimari_tipi
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        if mimari_tipi == "multi_head":
            # Sadece Multi-Head Attention (4 head x 8 = 32 boyut)
            self.sa = MultiHeadAttention(num_heads=4, head_size=8)
        elif mimari_tipi == "tam_blok":
            # Multi-Head + FeedForward + Residual + LayerNorm
            self.block = Block(n_embd=n_embd, n_head=4)
            self.ln_f = nn.LayerNorm(n_embd)
            
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        if self.mimari_tipi == "multi_head":
            x = self.sa(x)
        elif self.mimari_tipi == "tam_blok":
            x = self.block(x)
            x = self.ln_f(x)
            
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# -------------------------------------------------------------
# 4. EĞİTİM VE ADIM ADIM TABLO OLUŞTURMA
# -------------------------------------------------------------
def modeli_egit(mimari_adi):
    print(f"\n--> Model Eğitiliyor: {mimari_adi}...")
    model = GPTMini(mimari_tipi=mimari_adi).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    losses = estimate_loss(model)
    return losses['val']

if __name__ == '__main__':
    print("=" * 65)
    print(" GÖREV 1 & 2: TRANSFORMER BLOĞU PARÇALARININ GELİŞİMİ")
    print("=" * 65)

    # 1. Multi-head modelini eğitip kaydediyoruz
    val_multi = modeli_egit("multi_head")
    
    # 2. Tam bloğu (Multi-head + FeedForward + Residual + LayerNorm) eğitiyoruz
    val_tam_blok = modeli_egit("tam_blok")

    print("\n" + "=" * 65)
    print(" HAFTA 8 - ADIM ADIM GELİŞİM TABLOSU (VAL LOSS)")
    print("=" * 65)
    print(f" 1. Taban Bigram (Hafta 7 Görev 1)           : 2.4897")
    print(f" 2. Tek Head Self-Attention (Hafta 7 Görev 4): 2.4133")
    print(f" 3. Multi-Head Attention (4 Head)            : {val_multi:.4f}")
    print(f" 4. Tam Transformer Bloğu (+FFWD, Res, LN)   : {val_tam_blok:.4f}")
    print("=" * 65)