# --- GÖREV 5: TÜRKÇE VS İNGİLİZCE KARŞILAŞTIRMASI ---
import tiktoken

# 1. GPT-4 (cl100k_base) ile Karşılaştırma
enc = tiktoken.get_encoding("cl100k_base")

tr_paragraf = "Yapay zeka modellerinin metinleri nasıl parçalara ayırdığını ve bu parçaların dilin yapısını nasıl etkilediğini inceliyoruz."
en_paragraf = "We are examining how artificial intelligence models break texts into pieces and how these pieces affect the structure of language."

tr_tokens_gpt = enc.encode(tr_paragraf)
en_tokens_gpt = enc.encode(en_paragraf)

tr_kelime_sayisi = len(tr_paragraf.split())
en_kelime_sayisi = len(en_paragraf.split())

print("--- GPT-4 (cl100k_base) Sonuçları ---")
print(f"Türkçe   : {len(tr_tokens_gpt)} token | {tr_kelime_sayisi} kelime -> Kelime başına {len(tr_tokens_gpt)/tr_kelime_sayisi:.2f} token")
print(f"İngilizce: {len(en_tokens_gpt)} token | {en_kelime_sayisi} kelime -> Kelime başına {len(en_tokens_gpt)/en_kelime_sayisi:.2f} token")


# 2. Kendi BPE Tokenizer'ımız ile Türkçe Paragrafı Test Etme
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

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats.keys(), key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

# Türkçe eğitim metnimizi okuyup merge tablosunu çıkaralım
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

# Kendi tokenizer'ımızla test
kendi_tr_tokens = encode(tr_paragraf, merges)
print("\n--- Kendi Tokenizer'ımız (306 Vocab) ---")
print(f"Türkçe   : {len(kendi_tr_tokens)} token | Kelime başına {len(kendi_tr_tokens)/tr_kelime_sayisi:.2f} token")