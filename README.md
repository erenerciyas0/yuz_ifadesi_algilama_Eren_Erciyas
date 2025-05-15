# 🎭 Gerçek Zamanlı Duygu Tanıma Uygulaması

Bu projede, bir kişinin kamerası üzerinden alınan görüntüdeki yüz ifadesi analiz edilerek **duygusal durumu** tespit edilmektedir. Sistem, dört temel ifadeyi tanıyabilir:

- Mutlu 😊
- Üzgün 😢
- Kızgın 😠
- Şaşkın 😲

---

## 🧰 Kullanılan Araçlar

- **Python 3.10+**
- **Mediapipe** – yüz landmark verisi için
- **OpenCV** – kamera kontrolü ve görselleştirme
- **Pillow (PIL)** – Türkçe yazı ve emoji desteği
- **scikit-learn** – sınıflandırma modeli
- **pandas** – veri seti yönetimi

---

## ⚙️ Gerekli Kurulumlar

Aşağıdaki komutla tüm gerekli kütüphaneleri yükleyebilirsiniz:

```bash
pip install mediapipe opencv-python scikit-learn pandas pillow
````

---

## 📁 Dosya Yapısı

```
.
├── yuz_algila.py            # Veri toplama aracı
├── egitim.py                # Model eğitimi
├── yuz_algila_test.py       # Gerçek zamanlı ifade tanıma
├── veriseti.csv             # Toplanan yüz verileri
├── model.pkl                # Eğitilmiş model
├── rapor.pdf                # Proje raporu
├── face_landmarker_v2_with_blendshapes.task  # Mediapipe modeli
└── README.md                # Bu açıklama dosyası
```

---

## ▶️ Kodların Çalıştırılması

### 1. Veri Toplama

```bash
python yuz_algila.py
```

> `etiket = "happy"` gibi satırı değiştirerek her ifade için ayrı çalıştırın. İlk çalıştırmada `sutun_basliklarini_olustur()` fonksiyonu açık olmalı.

---

### 2. Model Eğitimi

```bash
python egitim.py
```

> Eğitilen model `model.pkl` dosyasına kaydedilir.

---

### 3. Gerçek Zamanlı Test

```bash
python yuz_algila_test.py
```

> Kamera açılır, tespit edilen yüz ifadesi Türkçe metin ve emoji ile birlikte ekranda gösterilir.

---

## 💡 Ekstra Özellikler

* `cv2.putText()` yerine `PIL.ImageDraw` kullanılmıştır, böylece:

  * **Türkçe karakterler (ş, ğ, ü...) doğru görünür**
  * **Emojiler düzgün şekilde yazılır**
* Yazılar konumsal olarak orta-yukarı bölüme yerleştirilmiştir.

---

## 👨‍🏫 Proje Bilgisi

Eren Erciyas
2023688028

```

