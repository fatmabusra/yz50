import urllib.request
import json
import os

print("Vikikaynak ve Türkçe klasik metinler derleniyor...")

# Türkçe Wikipedia'dan klasik ve kapsamlı edebi/tarihi başlıklar
basliklar = [
    "Mustafa_Kemal_Atatürk",
    "Nutuk_(Mustafa_Kemal_Atatürk)",
    "Kurtuluş_Savaşı",
    "Osmanlı_İmparatorluğu",
    "Dede_Korkut_Kitabı",
    "Divânu_Lugâti't-Türk",
    "Kutadgu_Bilig",
    "Yunus_Emre",
    "Mevlânâ_Celâleddîn-i_Rûmî",
    "Fuzûlî",
    "Evliya_Çelebi",
    "Türk_edebiyatı"
]

derlenen_metin = ""

for baslik in basliklar:
    url = f"https://tr.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={baslik}&format=json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req) as response:
            veri = json.loads(response.read().decode('utf-8'))
            pages = veri['query']['pages']
            for p_id in pages:
                if 'extract' in pages[p_id]:
                    derlenen_metin += pages[p_id]['extract'] + "\n\n"
        print(f"+ {baslik.replace('_', ' ')} metni eklendi.")
    except Exception as e:
        print(f"! {baslik} atlandı: {e}")

# Tiny Shakespeare ~1.1 MB (yaklaşık 1 milyon karakter) olduğu için 
# Metni hedef boyuta (~1 MB) ulaştıracak şekilde dengeli çoğaltıyoruz
hedef_karakter = 1_000_000
if len(derlenen_metin) > 0:
    tekrar_sayisi = (hedef_karakter // len(derlenen_metin)) + 1
    derlenen_metin = (derlenen_metin * tekrar_sayisi)[:hedef_karakter]

with open("turkce_input.txt", "w", encoding="utf-8") as f:
    f.write(derlenen_metin)

boyut_mb = os.path.getsize("turkce_input.txt") / (1024 * 1024)
print("-" * 55)
print(f"Tamamlandı! 'turkce_input.txt' başarıyla oluşturuldu.")
print(f"Dosya Boyutu: {boyut_mb:.2f} MB")
print(f"Toplam Karakter: {len(derlenen_metin):,}")