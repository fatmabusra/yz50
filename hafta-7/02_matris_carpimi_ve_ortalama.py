import torch
import torch.nn.functional as F

# -------------------------------------------------------------
# GÖREV 2: GEÇMİŞİN ORTALAMASINI ÜÇ FARKLI YOLDAN ALMA
# -------------------------------------------------------------
torch.manual_seed(1337)

# Basit bir deneme tensörü: 4 örnek (B), 8 harf (T), 2 özellik (C)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# =============================================================
# 1. YOL: DÜZ FOR DÖNGÜSÜ (EN İLKEL YOL)
# =============================================================
# Her cümlenin her harfinde geriye dönüp geçmişteki harflerin ortalamasını alırız.
xbow1 = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        # t anına kadar olan geçmiş harfleri kesip alıyoruz (0'dan t'ye kadar)
        gecmis = x[b, :t+1] # Şekil: (t+1, C)
        # Bu harflerin ortalamasını alıp yerine koyuyoruz
        xbow1[b, t] = torch.mean(gecmis, 0)

# =============================================================
# 2. YOL: torch.tril İLE MATRİS ÇARPIMI (HIZLI YOL)
# =============================================================
# Alt üçgen matris oluşturuyoruz (köşegen ve altı 1, üstü 0)
wei1 = torch.tril(torch.ones(T, T))

# Her satırı kendi toplamına bölüyoruz ki satır toplamı 1 olsun (ortalama ağırlığı)
wei1 = wei1 / wei1.sum(1, keepdim=True)

# Matris çarpımı yapıyoruz: wei1 @ x
# Bu çarpım, her harf için geçmişteki harflerin ağırlıklı ortalamasını tek seferde hesaplar!
xbow2 = wei1 @ x

# =============================================================
# 3. YOL: SOFTMAX VE -INF İLE (SELF-ATTENTION'A GİDEN YOL)
# =============================================================
# Alt üçgen maskemiz
tril = torch.tril(torch.ones(T, T))

# Başlangıçta tüm ağırlıklar 0
wei2 = torch.zeros((T, T))

# Gelecekteki harflerin olduğu yerleri -sonsuz (-inf) yapıyoruz
# Çünkü model henüz gelmemiş gelecekteki harfleri görmemeli!
wei2 = wei2.masked_fill(tril == 0, float('-inf'))

# Son boyutta softmax uyguluyoruz:
# -inf olan yerler 0 olur, 0 olan yerler ise eşit şekilde paylaştırılıp toplamı 1 yapar
wei2 = F.softmax(wei2, dim=-1)

# Matris çarpımıyla ortalamayı alıyoruz
xbow3 = wei2 @ x

# =============================================================
# KONTROL: ÜÇÜ DE AYNI ŞEYİ Mİ YAPTI?
# =============================================================
esit_mi_1_ve_2 = torch.allclose(xbow1, xbow2, atol=1e-6)
esit_mi_2_ve_3 = torch.allclose(xbow2, xbow3, atol=1e-6)

print("Ağırlık Matrisi wei (İlk 3 Satır):")
print(wei2[:3, :3])
print("-" * 55)
print(f"1. Yol ile 2. Yol aynı mı? : {esit_mi_1_ve_2}")
print(f"2. Yol ile 3. Yol aynı mı? : {esit_mi_2_ve_3}")