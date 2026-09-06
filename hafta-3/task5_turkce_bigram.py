import torch

# 1. Dosyayı oku
try:
    lines = open('turkce_isimler.txt', 'r', encoding='utf-8').read().splitlines()
except Exception:
    lines = open('turkce_isimler.txt', 'r', encoding='latin-1').read().splitlines()

# Sadece gerçek Türkçe küçük harfleri filtrele (rakam ve işaretleri at)
turkish_letters = set("abcçdefgğhıijklmnoöprsştuüvyz")
words = []
for line in lines:
    clean = line.strip().lower()
    # Sadece harflerden oluşan isimleri al
    if clean and all(ch in turkish_letters for ch in clean):
        words.append(clean)

# Eğer dosya boşsa ya da bulunamazsa yedek liste
if len(words) < 10:
    words = ["ahmet", "mehmet", "ayşe", "fatma", "büşra", "zeynep", "emre", "can", "oğuz", "çağrı", "şule", "özlem", "ümit", "ışıl", "burak", "deniz", "elif", "kaan", "selin", "mert", "ece", "tarık", "yusuf", "gamze", "tuğçe", "gökhan", "ceren", "murat", "serkan", "hakan"]

print("Kullanılan temiz Türkçe isim sayısı:", len(words))

# 2. Alfabeyi oluştur
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)
print(f"Alfabedeki karakter sayısı (. dahil): {vocab_size}")

# 3. Sayım tablosu
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# 4. Olasılık tablosu (Smoothing ile)
P = (N + 1).float()
P = P / P.sum(1, keepdim=True)

# 5. Türkçe NLL Loss hesabı
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob)
        n += 1

loss = -log_likelihood / n
print(f"Türkçe Veri Seti Loss (NLL): {loss.item():.4f}")

# 6. Türkçe İsim Üretme
g = torch.Generator().manual_seed(42)
print("\n--- Modelin Ürettiği Türkçe İsimler ---")
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out[:-1]))