import math
import torch
from gpt2_model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modeli rastgele parametrelerle başlat (Ağırlık yüklemeden)
config = GPTConfig()
model = GPT(config).to(device)

# Rastgele veri oluştur
x_dummy = torch.randint(0, 50257, (4, 32), device=device)
y_dummy = torch.randint(0, 50257, (4, 32), device=device)

# 1. İlk Loss Kontrolü
logits, loss_initial = model(x_dummy, y_dummy)
print(f"Rastgele Model İlk Loss: {loss_initial.item():.4f}")
print(f"Teorik Beklenen Loss (ln(50257)): {math.log(50257):.4f}\n")

# 2. Tek Batch Ezberletme (Overfitting Testi)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("--- Tek Batch Ezberletme Başlıyor ---")
for step in range(150):
    optimizer.zero_grad()
    logits, loss = model(x_dummy, y_dummy)
    loss.backward()
    optimizer.step()

    if step % 25 == 0 or step == 149:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")