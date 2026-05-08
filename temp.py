import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/Users/meltemcap/Projects/SocialMediaImpactonTeenMentalHealth/Teen_Mental_Health_Dataset.csv")

print("İlk 5 satır:")
df.head() 

print("\nVeri seti bilgisi:")
df.info()

print("\nİstatistiksel özet:")
df.describe()

print("\nHedef değişken dağılımı:")
df['depression_label'].value_counts()


# ============================================================
#  KEŞİFSEL VERİ ANALİZİ (EDA) 
# ============================================================


# Grafik 1: Hedef değişken dağılımı
plt.figure(figsize=(6, 4))
df['depression_label'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Depression Label Dağılımı')
plt.xlabel('Depression Label')
plt.ylabel('Sayı')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('grafik1_hedef_dagilim.png', dpi=150)
plt.show()


# Grafik 2: Sayısal değişkenlerin korelasyon ısı haritası
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Korelasyon Isı Haritası')
plt.tight_layout()
plt.savefig('grafik2_korelasyon.png', dpi=150)
plt.show()

# Grafik 3: Sosyal medya saati vs depression
plt.figure(figsize=(7, 5))
sns.boxplot(x='depression_label', y='daily_social_media_hours', data=df,
            palette='Set2')
plt.title('Günlük Sosyal Medya Saati vs Depression Label')
plt.xlabel('Depression Label')
plt.ylabel('Günlük Sosyal Medya Saati')
plt.tight_layout()
plt.savefig('grafik3_sosyal_medya_vs_depression.png', dpi=150)
plt.show()

# ============================================================
# 3. VERİ ÖN İŞLEME
# ============================================================

df_processed = df.copy()

# 3.1 Eksik veri kontrolü
print("\nEksik veri sayısı:")
print(df_processed.isnull().sum())
# Eksik veri yok → herhangi bir işlem gerekmez

# 3.2 Kategorik değişkenleri encode et (Label Encoding)
le = LabelEncoder()
categorical_cols = ['gender', 'platform_usage', 'social_interaction_level']

for col in categorical_cols:
    df_processed[col] = le.fit_transform(df_processed[col])
    print(f"{col} → encode edildi: {df[col].unique()} → {df_processed[col].unique()}")
    

# 3.3 Özellikler ve hedef değişkeni ayır
X = df_processed.drop('depression_label', axis=1)
y = df_processed['depression_label']

# 3.4 Normalizasyon (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nNormalizasyon sonrası istatistikler:")
print(pd.DataFrame(X_scaled).describe().round(2))

# 3.5 Train-Test Split (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nEğitim seti: {X_train.shape[0]} örnek")
print(f"Test seti:   {X_test.shape[0]} örnek")

# ============================================================
# 3.6 SMOTE - Veri Dengesizliği Giderme
# ============================================================
from imblearn.over_sampling import SMOTE

print("SMOTE ÖNCESİ eğitim seti dağılımı:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nSMOTE SONRASI eğitim seti dağılımı:")
print(pd.Series(y_train_sm).value_counts())

# SMOTE karşılaştırma grafiği
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (data, title) in zip(axes, [
    (y_train, 'SMOTE Öncesi'),
    (pd.Series(y_train_sm), 'SMOTE Sonrası')
]):
    counts = data.value_counts()
    ax.bar(counts.index.astype(str), counts.values, color=['#2ecc71', '#e74c3c'])
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Depression Label')
    ax.set_ylabel('Örnek Sayısı')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.suptitle('SMOTE: Veri Dengesizliği Giderme', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafik4_smote.png', dpi=150)
plt.show()

# ============================================================
# 4. RANDOM FOREST MODELİ
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_sm, y_train_sm)  # SMOTE'lu veriyle eğit
y_pred_rf = rf_model.predict(X_test)  # Orijinal test setiyle test et

print("\n========== RANDOM FOREST SONUÇLARI ==========")
acc = accuracy_score(y_test, y_pred_rf)
f1  = f1_score(y_test, y_pred_rf, average='weighted')
print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf,
      target_names=['Depresyon Yok (0)', 'Depresyon Var (1)']))

# 5-Fold Cross Validation
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"5-Fold CV: {[round(s,4) for s in cv_scores]}")
print(f"Ortalama : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, ax=ax,
    display_labels=['Depresyon Yok', 'Depresyon Var'],
    colorbar=False, cmap='Blues')
ax.set_title('Random Forest - Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('grafik5_confusion_matrix.png', dpi=150)
plt.show()

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importances_sorted.plot(kind='barh', color='steelblue')
plt.title('Random Forest - Özellik Önem Sıralaması', fontweight='bold')
plt.xlabel('Önem Skoru')
plt.tight_layout()
plt.savefig('grafik6_feature_importance.png', dpi=150)
plt.show()

# ============================================================
# 5. XGBOOST MODELİ - Karşılaştırma
# ============================================================
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(X_train_sm, y_train_sm)
y_pred_xgb = xgb_model.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb  = f1_score(y_test, y_pred_xgb, average='weighted')

print("\n========== XGBOOST SONUÇLARI ==========")
print(f"Accuracy : {acc_xgb:.4f}  ({acc_xgb*100:.2f}%)")
print(f"F1 Score : {f1_xgb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb,
      target_names=['Depresyon Yok (0)', 'Depresyon Var (1)']))

# XGBoost Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_xgb, ax=ax,
    display_labels=['Depresyon Yok', 'Depresyon Var'],
    colorbar=False, cmap='Oranges')
ax.set_title('XGBoost - Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('grafik_confusion_xgb.png', dpi=150)
plt.show()

# ============================================================
# 6. ROC CURVE - Her İki Model
# ============================================================

# Tahmin olasılıkları
y_prob_rf  = rf_model.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# AUC skorları
auc_rf  = roc_auc_score(y_test, y_prob_rf)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

# ROC eğrileri
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(7, 5))
plt.plot(fpr_rf,  tpr_rf,  label=f'Random Forest (AUC = {auc_rf:.4f})',
         color='steelblue', lw=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost      (AUC = {auc_xgb:.4f})',
         color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], '--', color='gray', label='Rastgele Tahmin')
plt.title('ROC Curve - Model Karşılaştırması', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('grafik_roc_karsilastirma.png', dpi=150)
plt.show()

print(f"\nRandom Forest AUC : {auc_rf:.4f}")
print(f"XGBoost AUC       : {auc_xgb:.4f}")

# ============================================================
# 7. ÖZET KARŞILAŞTIRMA TABLOSU
# ============================================================
print("\n" + "=" * 60)
print("ÖZET MODEL KARŞILAŞTIRMASI")
print("=" * 60)
print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10} {'AUC':>10}")
print("-" * 60)
print(f"{'Random Forest':<20} {accuracy_score(y_test, y_pred_rf):>10.4f} "
      f"{f1_score(y_test, y_pred_rf, average='weighted'):>10.4f} {auc_rf:>10.4f}")
print(f"{'XGBoost':<20} {acc_xgb:>10.4f} {f1_xgb:>10.4f} {auc_xgb:>10.4f}")