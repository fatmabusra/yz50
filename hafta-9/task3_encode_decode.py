# --- GÖREV 3: ENCODE VE DECODE ---

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# 1. Metni oku ve merges tablosunu eğit (task2'deki gibi)
with open("turkce_input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = list(text.encode("utf-8"))
vocab_size = 306
num_merges = vocab_size - 256
merges = {}

for i in range(num_merges):
    stats = get_stats(tokens)
    if not stats:
        break
    best_pair = max(stats, key=stats.get)
    new_id = 256 + i
    tokens = merge(tokens, best_pair, new_id)
    merges[best_pair] = new_id

# Vocab sözlüğünü oluştur
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


# 2. Decode ve Encode Fonksiyonları
def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    # Hatalı/geçersiz byte dizilimlerinde çökmeyi engellemek için errors='replace' kullanılır
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # Metindeki ikililerden merges tablosunda en erken öğrenileni bul
        pair = min(stats.keys(), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


# 3. Test Kısmı
test_metni = "Merhaba dünya! 🚀 Şampiyon çay içer."
token_ids = encode(test_metni, merges)
cozulmus_metin = decode(token_ids, vocab)

print(f"Orijinal Metin : {test_metni}")
print(f"Token ID'leri  : {token_ids}")
print(f"Geri Çözülen   : {cozulmus_metin}")

assert test_metni == cozulmus_metin, "Hata: Çözülen metin orijinaliyle aynı değil!"
print("Doğrulama Başarılı: decode(encode(metin)) == metin")