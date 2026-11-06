import torch
import torch.nn.functional as F
import tiktoken
from gpt2_model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

print("Ağırlıklar yükleniyor...")
model = GPT.from_pretrained("gpt2").to(device)
model.eval()

def generate(prompt, max_new_tokens=25, top_k=50):
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= 1024 else x[:, -1024:]
        with torch.no_grad():
            logits, _ = model(x_cond)
            logits = logits[:, -1, :]
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    return enc.decode(x[0].tolist())

print("\n--- İNGİLİZCE ÇIKTI ---")
print(generate("Hello, I'm a language model,"))

print("\n--- TÜRKÇE ÇIKTI ---")
print(generate("Bugün hava çok güzel ve ben"))

# Tokenizer Karşılaştırması
tr_text = "Bugün hava çok güzel ve ben"
en_text = "Today the weather is very nice and I"

tr_tokens = enc.encode(tr_text)
en_tokens = enc.encode(en_text)

print(f"\nTürkçe Cümle Token Sayısı: {len(tr_tokens)} -> {tr_tokens}")
print(f"İngilizce Cümle Token Sayısı: {len(en_tokens)} -> {en_tokens}")