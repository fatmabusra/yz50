import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------
# 1. VERİ VE HİPERPARAMETRELER
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

# Parametreler 
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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
        out[split] = losses.mean()
    model.train()
    return out

# -------------------------------------------------------------
# 2. TEK BİR SELF-ATTENTION HEAD MODÜLÜ
# -------------------------------------------------------------
class Head(nn.Module):
    """ Tek bir self-attention başlığı """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # İlişki skorları (Scaled Dot-Product)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Değerlerle ağırlıklı toplama
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

# -------------------------------------------------------------
# 3. SELF-ATTENTION İÇEREN DİL MODELİ
# -------------------------------------------------------------
class BigramWithSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd) # Tek bir self-attention head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_head(x)   # Self-attention uygula (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
            
            # girdiyi son block_size kadar kırpıyoruz!
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -------------------------------------------------------------
# 4. EĞİTİM VE KIYASLAMA
# -------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print(" GÖREV 4: TEK HEAD'Lİ SELF-ATTENTION MODELİ EĞİTİLİYOR")
    print("=" * 65)

    model = BigramWithSelfAttention().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Adım {iter:4d}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_losses = estimate_loss(model)
    
    print("\n" + "=" * 65)
    print(" VAL LOSS KARŞILAŞTIRMASI")
    print("=" * 65)
    print("1. Görev Taban Bigram Val Loss      : 2.4897")
    print(f"4. Görev Tek Head ile Val Loss      : {final_losses['val']:.4f}")
    fark = 2.4897 - final_losses['val'].item()
    print(f"Düşüş (İyileşme) Miktarı             : {fark:.4f} puan")
    print("=" * 65)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\nTek Head Modelinin Ürettiği Örnek Metin:")
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))