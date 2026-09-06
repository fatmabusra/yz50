import torch

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# +1 sahte sayım ekleyerek smoothing yapıyoruz
P = (N + 1).float()
P = P / P.sum(1, keepdim=True)

# Negative Log Likelihood hesabı:
# Olasılıkları çarpmak yerine logaritmalarını topluyoruz sayısal taşma olmasın diye.
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        
        # Logaritmasını alıp biriktiriyoruz
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

# Log negatif çıktığı için eksi ile çarpıp pozitife çeviriyoruz (Negative Log Likelihood)
nll = -log_likelihood

# Kelime sayısına bölerek ortalama loss değerini buluyoruz
loss = nll / n

print("Toplam bakılan harf ikilisi:", n)
print("Hesaplanan Loss (NLL):", loss.item())
print("Beklenen Karpathy skoru: ~2.4540")