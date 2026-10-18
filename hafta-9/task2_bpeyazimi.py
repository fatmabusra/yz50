# --- GÖREV 2: SIFIRDAN BPE EĞİTİMİ ---

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        # Eğer mevcut ve sonraki eleman hedef çifte eşitse birleştir
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# Geçen haftadan elindeki Türkçe metin dosyasını oku
with open("turkce_input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Metni ham byte listesine çevir (0 - 255 arası tamsayılar)
tokens = list(text.encode("utf-8"))
baslangic_byte_sayisi = len(tokens)

# Kaç adet birleştirme (merge) yapacağız?
# Örneğin 256 temel byte + 50 yeni token = 306 vocab boyutu
vocab_size = 306
num_merges = vocab_size - 256
merges = {} # (p0, p1) -> new_id

print(f"Başlangıçtaki token (byte) sayısı: {baslangic_byte_sayisi}")

for i in range(num_merges):
    stats = get_stats(tokens)
    if not stats:
        break
    best_pair = max(stats, key=stats.get)
    new_id = 256 + i
    tokens = merge(tokens, best_pair, new_id)
    merges[best_pair] = new_id

bitis_token_sayisi = len(tokens)
print(f"Eğitim sonrası token sayısı: {bitis_token_sayisi}")

# Vocab sözlüğünü oluşturalım: ID -> bytes
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

# İlk 20 birleştirmeyi yazdır
print("\n--- İlk 20 Birleştirme ---")
for i, ((p0, p1), idx) in enumerate(list(merges.items())[:20]):
    merged_bytes = vocab[idx]
    # Ekrana basarken hata vermemesi için replace kullanıyoruz
    metin_hali = merged_bytes.decode('utf-8', errors='replace')
    print(f"{i+1:2d}. Token {idx}: {repr(metin_hali)} (İkili: {p0} + {p1})")

sikistirma_orani = baslangic_byte_sayisi / bitis_token_sayisi
print(f"\nSıkıştırma Oranı (Byte / Token): {sikistirma_orani:.2f}X")