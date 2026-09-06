import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# 1. Eğitim veri setini hazırlıyoruz (x: girdi harf, y: hemen sonra gelen hedef harf)
xs = []
ys = []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = len(xs)
print("Örnek sayısı:", num)

# 2. Ağırlık matrisi (W): 27 girdi, 27 çıktı
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# 3. Gradient Descent ile Eğitim Döngüsü
for k in range(100):
    
    # Forward Pass
    # Girdi harflerini one-hot vektörüne çeviriyoruz
    xenc = F.one_hot(xs, num_classes=27).float()
    
    # Nöron çarpımı: x * W
    logits = xenc @ W
    
    # Softmax uyguluyoruz (önce e üzeri al, sonra satır toplamına böl)
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    
    # Doğru harflerin olasılıklarının negatif log ortalamasını alıyoruz (Loss)
    loss = -probs[torch.arange(num), ys].log().mean()
    
    # Backward Pass
    # Geçen hafta elle yazdığımız gradyan sıfırlama ve backward
    W.grad = None
    loss.backward()
    
    # Ağırlıkları eğim yönünün tersine adım attırıyoruz
    W.data += -50.0 * W.grad
    
    if (k + 1) % 20 == 0 or k == 0:
        print(f"Adım {k+1:03d} | Loss: {loss.item():.4f}")

print("\nFinal Loss:", loss.item())