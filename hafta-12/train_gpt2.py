import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from transformers import GPT2LMHeadModel
from hellaswag import render_example, iterate_examples

# -----------------------------------------------------------------------------
# MODEL MİMARİSİ
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
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
        return self.c_proj(self.gelu(self.c_fc(x)))

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
    vocab_size: int = 50304
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
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in torch.optim.AdamW.__doc__
        use_fused = fused_available and device_type == 'cuda'
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

    @classmethod
    def from_pretrained(cls, model_type='gpt2'):
        config = GPTConfig()
        config.vocab_size = 50257
        model = GPT(config)
        sd = model.state_dict()
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

# -----------------------------------------------------------------------------
# GÖREV 1: Shard Dosyalarından Okuyan DataLoaderLite
# -----------------------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = "edu_fineweb_mini"
        shards = sorted([os.path.join(data_root, s) for s in os.listdir(data_root) if split in s])
        self.shards = shards
        assert len(shards) > 0, f"{split} için shard dosyası bulunamadı!"
        print(f"DataLoader ({split}): {len(shards)} adet shard dosyası bulundu.")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------
# GÖREV 4: HellaSwag Değerlendirme Fonksiyonu
# -----------------------------------------------------------------------------
def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()

def evaluate_hellaswag(model, device, dtype, max_examples=1000):
    model.eval()
    num_correct_norm = 0
    num_total = 0
    with torch.no_grad():
        for i, example in enumerate(iterate_examples("validation")):
            if i >= max_examples:
                break
            tokens, mask, label = render_example(example)
            tokens, mask = tokens.to(device), mask.to(device)
            with torch.amp.autocast(device_type=device, dtype=dtype):
                logits, _ = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
    return num_correct_norm, num_total, (num_correct_norm / num_total)

# -----------------------------------------------------------------------------
# EĞİTİM DÖNGÜSÜ (GÖREV 2 & 3)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Cihaz: {device} | Hassasiyet: {dtype}")

    if device == "cuda":
        torch.set_float32_matmul_precision('high')

    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    total_batch_size = 8192
    B = 4
    T = 1024
    grad_accum_steps = total_batch_size // (B * T)

    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader = DataLoaderLite(B=B, T=T, split="val")

    model = GPT(GPTConfig()).to(device)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

    max_steps = 120
    warmup_steps = 15
    max_lr = 6e-4
    min_lr = max_lr * 0.1

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    train_losses, val_losses, val_steps = [], [], []
    total_tokens_processed = 0
    total_time_sec = 0.0

    print("\n--- Eğitim Başlıyor ---")
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % 25 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 10
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast(device_type=device, dtype=dtype):
                        logits, loss = model(x, y)
                    val_loss_accum += (loss / val_loss_steps).detach()
            print(f"\n[Adım {step:3d}] Validation Loss: {val_loss_accum.item():.4f}")
            val_losses.append(val_loss_accum.item())
            val_steps.append(step)

        if step in {0, 60} or last_step:
            model.eval()
            num_return_sequences = 3
            max_length = 32
            tokens = enc.encode("Hello, I'm a language model,")
            xgen = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)
            while xgen.size(1) < max_length:
                with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=dtype):
                    logits, _ = model(xgen)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                    xgen = torch.cat((xgen, torch.gather(topk_indices, -1, ix)), dim=1)
            print(f"--- Adım {step} Örnek Metin Üretimi ---")
            for i in range(num_return_sequences):
                print(f"Örnek {i+1}: {enc.decode(xgen[i, :max_length].tolist())}")
            print("--------------------------------------\n")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device, dtype=dtype):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        total_tokens_processed += tokens_processed
        total_time_sec += dt
        train_losses.append(loss_accum.item())

        if step % 10 == 0 or last_step:
            print(f"Step {step:3d} | Train Loss: {loss_accum.item():.4f} | Norm: {norm:.2f} | LR: {lr:.2e} | Süre: {dt*1000:.0f}ms | Tok/sec: {tokens_per_sec:.0f}")

    print(f"\nEğitim Tamamlandı! Toplam Görülen Token: {total_tokens_processed:,} | Ortalama Hız: {total_tokens_processed/total_time_sec:.0f} tok/sec | Son Train Loss: {train_losses[-1]:.4f} | Son Val Loss: {val_losses[-1]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(max_steps), train_losses, label="Train Loss", alpha=0.75)
    plt.plot(val_steps, val_losses, label="Validation Loss", marker="o", linewidth=2, color="red")
    plt.title("GPT-2 (124M) Eğitim ve Doğrulama Loss Eğrisi")
    plt.xlabel("Adım (Step)")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()
    print("Grafik 'loss_curve.png' olarak kaydedildi.")

    print("\n--- HellaSwag Değerlendirmesi Başlıyor (İlk 1.000 Örnek) ---")
    c_own, t_own, acc_own = evaluate_hellaswag(model, device, dtype, max_examples=1000)
    print(f"Bizim Eğittiğimiz Model HellaSwag Skoru: {c_own}/{t_own} = %{acc_own*100:.2f}")

    del model
    torch.cuda.empty_cache()

    print("OpenAI GPT-2 (124M) ağırlıkları yükleniyor...")
    openai_model = GPT.from_pretrained('gpt2').to(device)
    c_oa, t_oa, acc_oa = evaluate_hellaswag(openai_model, device, dtype, max_examples=1000)
    print(f"OpenAI GPT-2 (124M) HellaSwag Skoru:     {c_oa}/{t_oa} = %{acc_oa*100:.2f}")