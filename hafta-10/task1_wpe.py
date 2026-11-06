import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

# Hugging Face GPT-2 modelini yükle
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()

print("--- HUGGING FACE GPT-2 PARAMETRELERİ ---")
total_params = 0
for k, v in sd_hf.items():
    print(f"{k:45s} | {list(v.shape)}")
    total_params += v.numel()

print(f"\nToplam Parametre Sayısı: {total_params:,}")

# Positional Embedding (wpe) Görselleştirme
wpe = sd_hf["transformer.wpe.weight"].cpu().numpy()

plt.figure(figsize=(10, 6))
plt.imshow(wpe, aspect="auto", cmap="viridis")
plt.colorbar(label="Embedding Değeri")
plt.xlabel("Embedding Dim (768)")
plt.ylabel("Pos Context / Sequence Length (1024)")
plt.title("GPT-2 Positional Embedding (wpe) Matrix")
plt.savefig("wpe_visualization.png")
print("\nGrafik 'wpe_visualization.png' olarak kaydedildi.")