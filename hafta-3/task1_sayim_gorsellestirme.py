import torch
import matplotlib.pyplot as plt

# 1. İsimleri dosyadan satır satır okuyoruz
words = open('names.txt', 'r').read().splitlines()

print("İlk 3 isim:", words[:3])
print("Toplam isim:", len(words))

# 2. Önce Python sözlüğü (dict) ile harf ikililerini sayıp mantığı görelim
b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

print("\nEn çok geçen ilk 3 bigram:")
sorted_b = sorted(b.items(), key=lambda kv: -kv[1])
for k, v in sorted_b[:3]:
    print(k, "->", v)

# 3. Şimdi bunu 27x27 bir torch tensörüne dökelim
# Harfleri alfabetik dizip her birine numara veriyoruz (a:1, b:2 ... nokta ise 0)
chars = sorted(list(set(''.join(words))))
stoi = {}
stoi['.'] = 0
for i, s in enumerate(chars):
    stoi[s] = i + 1

# Numaradan harfe geri dönmek için ters sözlük
itos = {}
for s, i in stoi.items():
    itos[i] = s

# 27 satır 27 sütunluk sıfır matrisi açıyoruz
N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print("\nN matrisi oluşturuldu, boyutu:", N.shape)

# 4. Tabloyu çizdirme (Videodaki görsel)
plt.figure(figsize=(14, 14))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=7)
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray', fontsize=7)
plt.axis('off')
plt.savefig("bigram_matrisi.png")
print("Görsel 'bigram_matrisi.png' olarak kaydedildi!")