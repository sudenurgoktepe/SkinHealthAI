# 💖 Cilt Sağlığı Asistanı - AI Destekli Chatbot

> Türkçe NLP destekli, yapay zekâ tabanlı bir cilt hastalığı ve cilt problemi tahmin uygulaması 🧠 + 🎁 öneri motoru + 💬 interaktif chatbot arayüzü.

---

## 🧠 Proje Hakkında

Bu proje, cilt sorunlarını analiz ederek hem **teşhis** hem de **ürün önerisi** sunan interaktif bir Python uygulamasıdır. Kullanıcılar serbest metin şeklinde semptomlarını yazar, sistem ise iki farklı NLP modeliyle analiz yaparak cilt tipi, cilt hastalığı veya problemi tahmin eder.

### 🔍 Çift Modelli Mimari

| Model Türü        | Kullanım Alanı                   | Teknoloji              |
|-------------------|----------------------------------|------------------------|
| Keras (BiLSTM)    | Cilt hastalıkları + cilt tipi    | TensorFlow/Keras       |
| HuggingFace BERT  | Cilt problemi + cilt tipi        | Transformers/PyTorch   |

Her model, Türkçe veri setiyle sıfırdan eğitilmiştir. Kullanıcı dostu PyWebIO arayüzü sayesinde bu iki model arasında geçiş yapılabilir.

---

## 📦 Özellikler

- 🔁 Model geçiş butonları (Keras ↔️ BERT)
- 🧴 Cilt tipi ve problemi eşleştirmeli ürün öneri motoru
- 🧠 Güven skoru ile tahmin doğruluk bildirimi
- 📊 Sınıf bazlı doğruluk grafikleri ve confusion matrix’ler
- 🎨 CSS destekli hoş bir arayüz (chat baloncukları, ürün kutuları, emoji desteği)
- 🔐 Pickle ile kayıtlı `tokenizer`, `LabelEncoder`, `model` dosyaları

---

## 🗃️ Veri Seti Bilgisi

### 1. Orijinal Veri
- İngilizce dilinde
- Kolonlar: `Text`, `Disease name`

### 2. Türkçeleştirme ve Genişletme

Veri seti aşağıdaki adımlarla dönüştürüldü:

| Yeni Kolon            | Açıklama                                        |
|------------------------|-------------------------------------------------|
| `aciklama`             | Orijinal İngilizce metnin Türkçesi              |
| `hastalik`             | Hastalık adı (Türkçe)                           |
| `cilttipi`             | Metne göre uygun cilt tipi                      |
| `onerilen_urun`        | Hastalığa yönelik önerilen ürünler              |
| `cilt_onerilen_urun`   | Cilt tipine uygun ürün önerisi                  |

#### 👨‍🔬 Uygulanan İşlemler

- `ceviri.py`: İngilizce metinlerin Türkçeye çevrilmesi
- `csvbuyut.py`: Veri dengeleme & augmentation
- `kontrol.py`, `eklenti.py`: Manuel içerik düzeltmeleri ve kontrol

---

## 🎯 Örnek Kullanım

**Giriş:**
```
Cildim pul pul dökülüyor ve yanaklarımda kaşıntılı kızarıklık var.
```

**Çıktı (BERT Mode):**
```
🌡️ Problemin: pullanma (%81.00)
🧬 Cilt Tipin: kuru (%96.62)

🎁 Önerilen Ürün:
• hyaluronik asit serum
• yoğun nemlendirici krem
• bariyer onarıcı krem
```

---

## 🖥️ Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Uygulamayı Başlat

```bash
python app.py
```

Tarayıcıda `http://localhost:8080` adresini açarak sohbet arayüzüne erişebilirsiniz.

---

## 📈 Model Performansı

### 🏥 Hastalık Modeli (BiLSTM)
- Validation Accuracy: %86+
- Başarı grafiği ve sınıf bazlı doğruluk oranları görselleştirilmiştir.

### 🌸 Cilt Problemi Modeli (BERT)
- Test Accuracy: %94+
- Precision / Recall / F1-skor raporları detaylı sunulmuştur.

### 💧 Cilt Tipi Modelleri
- BERT ve BiLSTM için ayrı ayrı eğitildi
- Her sınıf için %100’e yakın doğruluk oranları elde edildi

---

## 📊 Görseller

> Aşağıdaki ekran görüntüleri veya GIF'ler eklenebilir:
> - Chatbot arayüzü (PyWebIO)
> - Model geçiş ekranı
> - Öneri kutuları
> - Confusion Matrix
> - Başarı oranı bar grafikleri

---

## 🛡️ Uyarı

> Bu araç **bir tıbbi teşhis sistemi değildir.** Sunulan öneriler yalnızca bilgilendirme amaçlıdır. Cilt sorunlarınız için dermatoloğa başvurunuz.

---

## 🤝 Katkıda Bulun

1. Forkla 🔀  
2. Geliştir 🔧  
3. Pull Request gönder 💌  

---

## 👨‍💻 Geliştirici

🧔 **Sudenur Göktepe**  
Bilişim Sistemleri Mühendisi ☁️  
GitHub: [github.com/kullaniciadi](https://github.com/sudenurgoktepe)

---

⭐ Eğer bu projeyi beğendiyseniz repoyu star'lamayı unutmayın! ⭐
