import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# -------------------------------------------------------------
# 1. GİRDİ VE POZİSYON EMBEDDING'İ HAZIRLIĞI
# -------------------------------------------------------------
B, T, C = 4, 8, 32   # Batch boyutu: 4, Bağlam uzunluğu: 8, Özellik boyutu: 32
head_size = 16       # Her bir head'in iç boyutu

# Rastgele bir girdi temsili (Batch=4, Zaman=8, Kanal=32)
x = torch.randn(B, T, C)

# Attention sırayı bilmediği için pozisyon embedding'i ekleriz
pos_emb_table = nn.Embedding(T, C)
pos = torch.arange(T) # [0, 1, 2, ..., 7]
pos_emb = pos_emb_table(pos) # (T, C)
x = x + pos_emb # Harf özellikleri + Pozisyon bilgisi: (B, T, C)

# -------------------------------------------------------------
# 2. TEK BİR SELF-ATTENTION HEAD (QUERY, KEY, VALUE)
# -------------------------------------------------------------
# Query: "Ben ne arıyorum?"
# Key:   "Bende ne bilgi var?"
# Value: "Eşleşirsek aktaracağım gerçek bilgi ne?"
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
v = value(x) # (B, T, 16)

# -------------------------------------------------------------
# 3. İLİŞKİ SKORLARI VE ÖLÇEKLENMİŞ MATRİS (WEI)
# -------------------------------------------------------------
# Query ile Key'in iç çarpımı: Harflerin birbiriyle alaka düzeyi
# (B, T, 16) @ (B, 16, T) -> (B, T, T)
wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)

# Gelecekteki harfleri görmemek için lower triangular maskesi
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

# Ağırlıklarla gerçek değerleri (Value) topluyoruz
out = wei @ v # (B, T, 16)

# -------------------------------------------------------------
# ÇIKTI 1: BİR ATTENTION SATIRININ ANALİZİ
# -------------------------------------------------------------
print("=" * 65)
print(" GÖREV 3A: WEI MATRİSİ VE BİR SATIRIN OKUNMASI")
print("=" * 65)
print("Örnek 0 için Ağırlık Matrisi wei (8x8):")
print(torch.round(wei[0], decimals=3))

print("\n--- 4. Satırın (İndeks 4 -> 5. Harf) Ağırlık Dağılımı ---")
satir_4 = wei[0, 4].tolist()
for i, agirlik in enumerate(satir_4):
    durum = "Gelecek (Maskeli)" if i > 4 else f"Geçmiş harf {i}"
    print(f"Harf {i:1d} payı: {agirlik:.4f}  ({durum})")

# -------------------------------------------------------------
# ÇIKTI 2: KAREKÖKE BÖLME (SCALING) DENEYİ
# -------------------------------------------------------------
print("\n" + "=" * 65)
print(" GÖREV 3B: SKORLARI HEAD_SIZE KAREKÖKÜNE BÖLME DENEYİ")
print("=" * 65)

# Rastgele ham skorlar üretelim (Varyansı head_size kadar büyük olsun)
ham_skorlar = torch.randn(8) * (head_size ** 0.5) # Varyansı ~16 civarında
olcekli_skorlar = ham_skorlar / (head_size ** 0.5) # Varyansı 1'e çekildi

# Softmax uyguluyoruz
softmax_bolunmemis = F.softmax(ham_skorlar, dim=-1)
softmax_bolunmus = F.softmax(olcekli_skorlar, dim=-1)

print("Ham Skorlar (Bölünmemiş)       :", [round(s, 2) for s in ham_skorlar.tolist()[:4]], "...")
print("Ölçekli Skorlar (Bölünmüş)      :", [round(s, 2) for s in olcekli_skorlar.tolist()[:4]], "...")
print("-" * 65)
print("Softmax Çıktısı (BÖLÜNMEDEN)    :", [round(p, 4) for p in softmax_bolunmemis.tolist()[:4]], "...")
print("Softmax Çıktısı (BÖLÜNEREK)     :", [round(p, 4) for p in softmax_bolunmus.tolist()[:4]], "...")
print("=" * 65)