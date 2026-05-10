## Social Media Impact on Teen Mental Health — Sınıflandırma Çalışması

## Proje Hakkında
Ergenlerin sosyal medya kullanım alışkanlıkları ve yaşam tarzı verilerine dayanarak
ruh sağlığı durumlarının sınıflandırılması amaçlanmıştır.

---

## Veri Seti
- **Kaynak:** Kaggle
- **Link:** https://www.kaggle.com/datasets/algozee/teenager-menthal-healy
- **Dosya:** `Teen_Mental_Health_Dataset.csv`
- **Görev Türü:** Gözetimli Öğrenme — Sınıflandırma (Supervised Learning — Classification)

---

## Kullanılan Teknolojiler
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---

## Uygulanan Adımlar

### 1. Veri Keşfi (EDA)
- Veri seti boyutu ve değişken tipleri incelendi
- İstatistiksel özet çıkarıldı
- Eksik veri analizi yapıldı
- Hedef değişken dağılımı görselleştirildi

### 2. Veri Ön İşleme
- Eksik değerler temizlendi
- Kategorik değişkenler encode edildi
- %80 eğitim / %20 test olarak ayrıldı
- StandardScaler ile normalizasyon uygulandı

### 3. Modelleme
Üç farklı sınıflandırma algoritması karşılaştırmalı olarak uygulanmıştır:

| Model | Açıklama |
|---|---|
| Logistic Regression | Baseline model |
| Random Forest | Topluluk tabanlı model |
| XGBoost | En iyi performans modeli |

### 4. Model Değerlendirme
- Accuracy, Precision, Recall, F1-Score metrikleri hesaplandı
- Confusion Matrix görselleştirildi
- ROC-AUC eğrisi çizildi

---

##  Temel Bulgular
- Sosyal medyada geçirilen süre ruh sağlığı üzerinde belirleyici bir etkiye sahiptir
- Uyku düzeni ve fiziksel aktivite ruh sağlığını doğrudan etkileyen faktörler arasındadır
- Erken teşhis için makine öğrenmesi modelleri umut verici sonuçlar ortaya koymaktadır

---

##  Toplumsal Önemi
Ergen ruh sağlığı günümüzün en kritik toplumsal sorunlarından biridir.
Bu çalışma sosyal medya kullanımının ergen psikolojisi üzerindeki etkisini
veri bilimi perspektifinden ele alarak erken müdahale için bir temel oluşturmayı
hedeflemektedir.
