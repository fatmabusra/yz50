import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------
# 1. VERİ HAZIRLIĞI VE TÜRKÇE VOCAB ANALİZİ
# -------------------------------------------------------------
# 1. Shakespeare Karakter Sayısı
with open('input.txt', 'r', encoding='utf-8') as f:
    sh_text = f.read()
sh_chars = sorted(list(set(sh_text)))
sh_vocab_size = len(sh_chars)

# 2. Türkçe Metin Yükleme
with open('turkce_input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print("=" * 65)
print(" GÖREV 4: KARAKTER KÜMESİ (VOCAB) KARŞILAŞTIRMASI")
print("=" * 65)
print(f"Tiny Shakespeare Vocab Boyutu : {sh_vocab_size} karakter")
print(f"Türkçe Metin Vocab Boyutu     : {vocab_size} karakter")
print(f"Fark                          : Türkçe alfabede ç, ğ, ı, ö, ş, ü gibi ek harfler")
print(f"                                ve noktalama işaretleri yer alır.")
print("=" * 65)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Görev 3 ile birebir aynı mimari parametreleri
batch_size = 32
block_size = 64
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

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
# 2. TRANSFORMER MİMARİSİ
# -------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TurkceGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -------------------------------------------------------------
# 3. TÜRKÇE MODELİ EĞİTME
# -------------------------------------------------------------
if __name__ == '__main__':
    model = TurkceGPT().to(device)
    param_sayisi = sum(p.numel() for p in model.parameters())

    print("\nTürkçe GPT Modeli Eğitiliyor...")
    print(f"Toplam Parametre: {param_sayisi:,}")
    print("-" * 65)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    baslangic = time.time()

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Adım {iter:4d}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    toplam_sure = time.time() - baslangic
    final_losses = estimate_loss(model)

    print("\n" + "=" * 65)
    print(" GÖREV 4 EĞİTİM SONUCU")
    print("=" * 65)
    print(f"Eğitim Süresi : {toplam_sure:.2f} saniye ({toplam_sure/60:.2f} dakika)")
    print(f"Son Val Loss  : {final_losses['val']:.4f}")
    print("=" * 65)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- TÜRKÇE GPT'NİN ÜRETTİĞİ ÖRNEK METİN ---")
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))