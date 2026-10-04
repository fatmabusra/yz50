import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------------------------------------------
# 1. VERİYİ OKUMA VE TOKENIZER'I KURMA (GÖREV 1)
# -------------------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Karakter kümesini ve sözlükleri oluştur
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Veriyi tensöre çevirme ve Train (%90) / Val (%10) olarak ayırma
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Hiperparametreler 
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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
# 2. TEMEL BIGRAM MODELİ (nn.Module Tabanlı)
# -------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Sadece bir karakterden bir sonrakini tahmin eden lookup tablosu
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx ve targets: (B, T) tensörleri
        logits = self.token_embedding_table(idx) # (B, T, C)

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
            logits, _ = self(idx)
            logits = logits[:, -1, :] # Son zamana odaklan (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -------------------------------------------------------------
# 3. EĞİTİM DÖNGÜSÜ VE TABAN LOSS KAYDI
# -------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print(" GÖREV 1: TİNY SHAKESPEARE BİGRAM MODELİ EĞİTİLİYOR")
    print(f" Toplam Karakter: {len(text)}, Sözlük Boyutu: {vocab_size}")
    print("=" * 60)

    model = BigramLanguageModel(vocab_size).to(device)
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
    print("=" * 60)
    print(f"BİGRAM TABAN VAL LOSS: {final_losses['val']:.4f}")
    print("=" * 60)

    # Örnek Üretim
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Bigram ile Üretilen Örnek Metin:")
    print(decode(model.generate(context, max_new_tokens=150)[0].tolist()))