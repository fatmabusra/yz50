import torch

words = open('names.txt', 'r').read().splitlines()

# Harf haritası
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Sayım matrisi
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# Sayımları olasılığa çevireceğiz
# +1 ekliyoruz çünkü hiç geçmeyen harfe 0 olasılık gelmesin (smoothing)
P = (N + 1).float()


# keepdim=True yazmazsak toplamın boyutu (27,) olur ve PyTorch bunu sütunlara böler.
# keepdim=True yazınca (27, 1) olur ve her satır kendi toplamına doğru bölünür.
P = P / P.sum(1, keepdim=True)

# Şimdi modelden rastgele isim türetelim
g = torch.Generator().manual_seed(2147483647)

print("--- Modelin Ürettiği İsimler ---")
for i in range(5):
    out = []
    ix = 0  # Başlangıç noktası '.'
    
    while True:
        p = P[ix]  # O anki harfin satırındaki olasılıklar
        
        # Olasılığa göre sıradaki harfi seçiyoruz
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        
        # Eğer tekrar '.' gelirse isim bitti demektir
        if ix == 0:
            break
            
    # Sonundaki noktayı yazdırmadan ekrana basalım
    print(''.join(out[:-1]))