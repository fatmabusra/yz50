import math
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# -----------------------------------------------------------------------------
# GÖREV 1: Data Loader (Metni Batch'lere Bölen Yapı)
# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        # Basit bir örnek metin (Gerçek senaryoda tinyshakespeare veya kendi metnin)
        enc = tiktoken.get_encoding("gpt2")
        # Örnek uzun metin simülasyonu
        text = "Hello, I am a language model. " * 1000
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"DataLoader: Toplam {len(self.tokens)} token yüklendi.")
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------
# GÖREV 1 & 2: Flash Attention Kullanan CausalSelfAttention
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Residual katman başlatma ölçeklemesi için bayrak
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # PyTorch Flash Attention (F.scaled_dot_product_attention)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # Görev 2: 50.257 yerine 2'nin kuvveti olan 50.304 (GPU padding hızı)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # GÖREV 1: Weight Tying
        self.lm_head.weight = self.transformer.wte.weight

        # GÖREV 1: Başlangıç Değerlerini Ayarlama (Init Weights)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Residual katmanlarda std = 0.02 / sqrt(2 * n_layer) ölçeklemesi
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # GÖREV 3: Weight Decay Seçici (Yalnızca 2D Parametrelere Uygulama)
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Weight Decay uygulanan (2D+) parametreler: {len(decay_params)} tensor, toplam {num_decay_params:,} parametre")
        print(f"Weight Decay uygulanmayan (1D) parametreler: {len(nodecay_params)} tensor, toplam {num_nodecay_params:,} parametre")

        fused_available = 'fused' in torch.optim.AdamW.__doc__
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
# GÖREV 3: Cosine Learning Rate Schedule ve Çizimi
# -----------------------------------------------------------------------------
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# LR Eğrisini Çizip Kaydetme
lrs = [get_lr(i) for i in range(max_steps)]
plt.figure(figsize=(8, 4))
plt.plot(lrs)
plt.title("GPT-3 Cosine Learning Rate Schedule with Warmup")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.savefig("lr_schedule.png")
print("Learning rate eğrisi 'lr_schedule.png' olarak kaydedildi.")

# -----------------------------------------------------------------------------
# EĞİTİM VE HIZLANDIRMA DÖNGÜSÜ (GÖREV 2 & 3)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan Cihaz: {device}")

    # GÖREV 2: Precision Ayarları
    if device == "cuda":
        torch.set_float32_matmul_precision('high')  # TF32 Aktivasyonu

    config = GPTConfig()
    model = GPT(config).to(device)

    # GÖREV 2: torch.compile (Donanım/OS Destekliyorsa)
    # try:
    #     model = torch.compile(model)
    #     print("torch.compile başarıyla aktifleştirildi.")
    # except Exception as e:
    #     print(f"torch.compile atlandı: {e}")

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
    train_loader = DataLoaderLite(B=4, T=1024)

    # GÖREV 3: Gradient Accumulation Ayarları
    total_batch_size = 16384  # ~16k token target batch size
    B = 4
    T = 1024
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Gradient Accumulation Adım Sayısı: {grad_accum_steps}")

    print("\n--- Eğitime Başlanıyor ---")
    for step in range(10):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # Mixed Precision (Autocast)
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast(device_type=device, dtype=dtype):
                logits, loss = model(x, y)
                # Loss'u birikim adımına bölüyoruz
                loss = loss / grad_accum_steps

            loss_accum += loss.detach()
            loss.backward()

        # GÖREV 3: Gradient Clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR Güncelleme
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000  # ms
        tokens_per_sec = (B * T * grad_accum_steps) / (t1 - t0)
        print(f"Step {step:2d} | Loss: {loss_accum.item():.4f} | Grad Norm: {norm:.4f} | LR: {lr:.2e} | Süre: {dt:.2f}ms | Tok/sec: {tokens_per_sec:.2f}")