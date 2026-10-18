# Hafta 9: Byte Pair Encoding (BPE) Tokenizer

Bu hafta büyük dil modellerinin metinleri nasıl okuduğunu anlamak için metni harf harf işlemek yerine GPT-4'ün kullandığı Byte Pair Encoding (BPE) yöntemini sıfırdan yazdık ve kendi Türkçe metnimiz üzerinde eğittik.

---

## 📌 Görev 1: Unicode, UTF-8 ve Kod Noktaları

Türkçe harfler (`ç, ğ, ı, ö, ş, ü`) içeren bir cümlenin kod noktalarını ve UTF-8 karşılıklarını inceledik:

- Standart İngilizce harfler ve boşluklar ASCII tablosunda yer aldığı için UTF-8'de **1 byte** yer tutar.
- Türkçe karakterler (`ç, ğ, ı, ö, ş, ü`) ise Unicode standardında 127'den büyük değerlere sahip olduğu için UTF-8'de **2 byte** yer kaplar.

### Kod Noktalarını Doğrudan Vocab Yapmak Neden Kötü Bir Fikir?
Unicode havuzunda 150 binden fazla karakter (tüm diller, semboller ve emojiler) vardır. Eğer her karaktere ayrı bir token verirsek:
1. Modelin kelime dağarcığı (vocab) 150 binden fazla olur. Bu da modelin giriş ve çıkış katmanlarını aşırı büyütür, bilgisayarın belleğini tüketir.
2. Yeni bir emoji veya bilinmeyen bir harf geldiğinde model bunu hiç tanıyamaz (Out-of-Vocabulary hatası verir).
Bunun yerine byte seviyesinde (0–255 arası) başlarsak, temel vocab sadece 256 olur ve dünyadaki bütün metinler sıfır kayıpla temsil edilebilir.

---

## 📌 Görev 2: Sıfırdan BPE Eğitimi ve Sıkıştırma

Türkçe metnimiz üzerinde ardışık ikilileri saydık (`get_stats`) ve en çok tekrar eden çiftleri yeni token ID'leri ile birleştirdik (`merge`).

- **Elde Edilen Sıkıştırma Oranı:** 1.47X (Metin byte seviyesinden token seviyesine geçince boyutu yaklaşık 1.5 kat küçüldü).

### İlk 20 Birleştirme (Merges)
Metinde ilk olarak Türkçede en sık kullanılan heceler ve kelime başındaki boşluklar birleşti:
1. `Token 256`: `' '` (boşluk ikilisi)
2. `Token 257`: `'er'`
3. `Token 258`: `' d'`
4. `Token 259`: `' b'`
5. `Token 260`: `'da'`
*(Diğer ilk 20 merge: `'ir'`, `'in'`, `'ıl'`, `'un'`, `'il'`, `'an'`, vb.)*

---

## 📌 Görev 3: Encode ve Decode Fonksiyonları

- `encode` fonksiyonu verilen metni önce byte'lara çevirir, sonra öğrendiğimiz merge kurallarına göre birleştirir.
- `decode` fonksiyonu bu token'ları tekrar byte'lara çevirip metin haline getirir.
- Türkçe karakterler ve roket emojisi (`🚀`) içeren test cümlemizde `decode(encode(metin)) == metin` eşitliği test edilmiş ve kayıpsız çalıştığı doğrulanmıştır.

### Geçersiz UTF-8 Durumunda Decode Ne Yapmalı?
BPE harf sınırlarına değil sadece byte'lara bakar. Model metin üretirken bazen 2 byte'lık bir Türkçe harfin sadece ilk byte'ını üretip durabilir. Bu durumda doğrudan `decode('utf-8')` çağrılırsa program hata verip çöker. Çökmeyi önlemek için `errors='replace'` parametresi kullanılır; böylece bozuk byte yerine ekrana  (yedek karakter) basılır ve kod çalışmaya devam eder.

---

## 📌 Görev 4: GPT-4 Regex Deseni

GPT-4'ün kullandığı regex desenini Türkçe kelimeler üzerinde denedik:
- `"Ankara'dan"` ➡️ `['Ankara', "'d", 'an']`
- `"İstanbul'da"` ➡️ `['İstanbul', "'d", 'a']`
- `"Ahmet'in"` ➡️ `['Ahmet', "'in"]`

### Desen Türkçede Ne Yapıyor?
Bu regex kuralı İngilizcedeki kısaltmalar (`don't`, `they'll`, `I'd`) için hazırlanmıştır. Kesme işaretinden sonra gelen harfleri İngilizce ekler sanır. Bu yüzden Türkçede `"Ankara'dan"` veya `"İstanbul'da"` yazdığımızda, `'d'` harfini İngilizcedeki `'d` sanarak kelimeyi kökünden ve ekinden koparıp yapay bir şekilde üçe böler. Yani Türkçe ek yapısına uymayan parçalamalar üretir.

---

## 📌 Görev 5: Türkçe ve İngilizce Karşılaştırması

Aynı anlama gelen cümleleri OpenAI'ın `cl100k_base` (GPT-4) tokenizer'ı ile test ettik:

- **İngilizce:** 20 kelime için 21 token harcandı ➡️ **Kelime başına 1.05 token**
- **Türkçe:** 15 kelime için 46 token harcandı ➡️ **Kelime başına 3.07 token**

### Türkçenin Eklemeli Yapısı Neden Bu Sonucu Veriyor?
GPT-4 çoğunlukla İngilizce metinlerle eğitildiği için çoğu İngilizce kelime hafızasında tek bir parça olarak durur. Ancak Türkçede kelimelerin sonuna çok sayıda ek geldiği için (`parçala-r-a`, `etki-le-diğ-i-ni`) model bu kelimeleri tek parça olarak tanıyamaz ve 2-3 parçaya bölmek zorunda kalır. Buna literatürde "Token Vergisi" denir; aynı anlamı ifade etmek için Türkçede 3 kat daha fazla token harcanır ve modelin hafızası daha çabuk dolar.

---

## 📌 Görev 6: Ek Görev - Val Loss Kıyaslama Analizi

> **Soru:** Tokenizer kullanan bir modelin validation loss değerini, harf seviyesindeki modelinkiyle doğrudan karşılaştırabilir miyiz?

**Cevap: Hayır, doğrudan karşılaştıramayız.**

1. **Vocab Boyutları Farklıdır:** Cross-entropy loss formülü doğrudan kelime havuzundaki seçenek sayısına bakar. Harf modelinde seçenek sayısı ~100 olduğu için modelin başlangıç loss'u düşüktür ($\approx 4.6$). Token modelinde ise vocab 300 veya 1000 olduğu için başlangıç loss'u çok daha yüksek başlar. İki model aynı zorlukta bir sınavda yarışmaz.
2. **Adım Başına Taşınan Bilgi Farklıdır:** Harf seviyesindeki model her tahminde sadece tek bir harf tahmin eder. Tokenizer kullanan model ise tek bir tahminde 3-4 harften oluşan bir heceyi veya kelimeyi tek seferde tahmin eder. Yani token modelinin tek bir adımı, harf modelinin 3-4 adımına eşdeğer bilgi (entropi) taşır.
3. **Nasıl Kıyaslanır?** İki modeli adil kıyaslamak için loss değerleri doğrudan değil, **Bits Per Byte (BPB)** yani harcanan byte başına düşen başarı metriğine çevrilerek karşılaştırılmalıdır.