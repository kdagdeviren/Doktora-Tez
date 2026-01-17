# ============================================================================
# MODEL ALL - LightGBM + SMOTE YOK İÇİN GELİŞMİŞ DOĞRULAMA ANALİZLERİ
# ============================================================================
# Bu scriptin amacı: daha önce “Model ALL + LightGBM + SMOTE YOK” kombinasyonunda
# bazı metriklerin (özellikle RCB-1 AUC=1.000 gibi) şüpheli derecede iyi görünmesi
# durumunda, modeli daha katı ve klinik açıdan anlamlı doğrulama araçlarıyla test etmek.
#
# Bu dosyada hedeflenen analizler:
# 1) Nested CV (overfitting/leakage riskini daha iyi yakalamak için)
# 2) Sınıf bazlı bootstrap CI (belirsizlik aralığı göstermek için)
# 3) Decision Curve Analysis (DCA) (klinik net fayda için)
# 4) Calibration analizi (Raw vs Platt vs Isotonic)
# =============================================================================

# ---------------------------
# 0) Genel ayarlar
# ---------------------------

import os, warnings, json
# warnings.filterwarnings('ignore'):
# Colab çıktısını temiz tutmak için (özellikle LightGBM/Sklearn uyarıları çok olur).
# Tez/GitHub açısından kapatılıp uyarıları incelemek daha şeffaf olabilir.
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
# files.upload():
# Colab ortamında kullanıcıdan dosya yükletmek için.

from sklearn.model_selection import train_test_split, StratifiedKFold
# train_test_split: önce train/test ayırarak data leakage riskini düşürür.
# StratifiedKFold: fold’larda sınıf oranlarını korur (RCB sınıfları dengesiz olabilir).

from sklearn.base import clone
# clone: her eğitimde temiz bir model instance’ı üretmek için.
# Nested CV içinde çok kritik: aynı model nesnesi tekrar fit edilirse state taşınabilir.

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, brier_score_loss
)
# brier_score_loss:
# Olasılık kalibrasyon kalitesini ölçmek için kullanılır.

from sklearn.preprocessing import label_binarize
# label_binarize:
# Multiclass’i one-vs-rest (OVR) binary forma çevirip sınıf bazında AUC hesaplamak için.

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
# calibration_curve: reliability diagram için
# CalibratedClassifierCV: Platt(sigmoid) ve Isotonic kalibrasyon için

from lightgbm import LGBMClassifier
# LightGBM: ağaç tabanlı boosting modeli; ölçekleme gerekmez.

from imblearn.over_sampling import SMOTE
# Not: Bu script “SMOTE YOK” odaklı.
# Ancak import edilmiş: ileride kıyas yapmak istersen (SMOTE VAR) kolay eklensin diye.
# (Bu scriptte fiilen SMOTE kullanılmıyor.)

import joblib
# joblib: model kaydetme için tipik (bu scriptte doğrudan kullanılmıyor ama ileride lazım olabilir).

# =============================================================================
# 1) VERİ YÜKLEME VE HAZIRLIK
# =============================================================================

print("=== VERİ YÜKLEME ===")
print("Lütfen Excel dosyanızı yükleyin:")

# uploaded: Colab arayüzünden dosya seçtirir; sözlük olarak döner.
uploaded = files.upload()

# Yüklenen dosya adını sözlüğün ilk anahtarından alıyoruz.
file_name = list(uploaded.keys())[0]

# Excel’i pandas DataFrame’e oku
data = pd.read_excel(file_name)

# target: çok-sınıflı hedef etiket kolon adı (0,1,2,3)
target = 'RCB_Kategorize'

# classes: sınıf sırasını sabitlemek için kullanılır.
# Özellikle ROC/AUC, binarize, plot sıralarında deterministik davranış sağlar.
classes = np.array([0, 1, 2, 3])

# ---------------------------
# 1.1) Model ALL feature setinin tanımı
# ---------------------------
# Bu çalışmada i11 ve i20 bilinçli olarak dışarıda bırakılmış.
# - i11 (E-cadherin) ve i20 (Menopoz) dışlanma gerekçesi: önceki ana projedeki karar.
# Model ALL: tüm grupların birleşimi (P+O+D+K+B+R)

features_p = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i12']
features_o = ['i13', 'i14', 'i15', 'i46', 'i47']
features_d = ['i16', 'i17', 'i18', 'i19', 'i45']
features_k = ['i21','i22','i23','i24','i25','i26','i27','i28','i29','i30']
features_b = ['i31','i32','i33','i34','i35','i36','i37','i38','i39','i40','i41','i42','i43','i44']
features_r = ['i48','i49','i50','i51','i52','i53','i54','i55','i56','i57','i58','i59','i60','i61','i62','i63','i64']

# Model ALL: tüm feature gruplarının birleşimi
feats_all = features_p + features_o + features_d + features_k + features_b + features_r

# X_all: sadece seçilen feature’lardan oluşan tasarım matrisi
X_all = data[feats_all].copy()

# y_all: hedef vektörü
y_all = data[target].copy()

# ---------------------------
# 1.2) Train/Test split (SMOTE YOK)
# ---------------------------
# Bu split kritik:
# - Tüm ileri analizler (nested CV, calibration, bootstrap) train/test ayrımına dayanır.
# - Test set, "sonuç raporlama" için bağımsız kalır.
# - Bu scriptte SMOTE uygulanmayacağı için, test set dağılımı doğrudan gerçek dünyayı yansıtır.
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    stratify=y_all,       # sınıf oranları train/test’te benzer kalsın
    random_state=42       # tekrarlanabilirlik
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Sınıf oranları (train/test):")
print(y_train.value_counts(normalize=True).sort_index())
print(y_test.value_counts(normalize=True).sort_index())

# ---------------------------
# 1.3) Model tanımı: LightGBM (SMOTE YOK)
# ---------------------------
# Bu hiperparametreler:
# - Daha önce ana analizde kullanılanlarla uyumlu olacak şekilde seçilmiş.
# - Amaç: “aynı model” üzerinde daha sıkı validasyon yapmak.
model = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# ---------------------------
# 1.4) Baseline: Train ile fit → Test’te tahmin
# ---------------------------
# Not:
# Nested CV bağımsız bir değerlendirme sağlayacak olsa da,
# bootstrap, DCA ve calibration analizi burada üretilen test olasılıklarını kullanıyor.
#
# Yani bu adım:
# - “Final modelin test üzerindeki olasılıklarını” üretmek için gerekli.
# - DCA ve calibration doğrudan olasılık kalitesine baktığı için predict_proba zorunlu.
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)
# y_proba_test şekli: (n_samples, 4)
# Her satır 4 sınıf için olasılık dağılımıdır; toplamı 1’e yakın olmalıdır.

# =============================================================================
# 2) NESTED CV - OVERFITTING KONTROLÜ
# =============================================================================
# Nested CV neden gerekli?
#
# Standart CV:
#   - Modeli seçer
#   - Performansı raporlar
#
# Nested CV:
#   - Model seçimi (inner loop)
#   - Performans değerlendirmesi (outer loop)
#   işlemlerini AYIRIR.
#
# Böylece:
# - Aşırı iyimser (optimistic bias) skorlar yakalanır
# - Özellikle “AUC=1.000” gibi sonuçlar sorgulanabilir hale gelir
#
# Bu çalışmada:
# - Outer CV: 5-fold (gerçek test simülasyonu)
# - Inner CV: 5-fold (modelin kendi iç kararlılığı)
# - SMOTE YOK (orijinal dağılım korunuyor)

print("\n=== NESTED CV ANALİZİ ===")
print("Dış loop: 5-fold, İç loop: 5-fold (toplam 25 model)")
print("Bu, overfitting kontrolü için daha katı bir değerlendirme sağlar.\n")

# ---------------------------
# 2.1) Outer ve Inner CV tanımları
# ---------------------------

# Outer CV:
# - “Mini test setleri” üretir
# - Modelin genellenebilirliğini ölçer
outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Inner CV:
# - Outer train içinde çalışır
# - Modelin kendi iç tutarlılığını ölçer
inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ---------------------------
# 2.2) Sonuçları saklamak için yapı
# ---------------------------
# Burada her sınıf için ayrı ayrı:
# - Inner CV AUC ortalaması
# - Inner CV AUC std
# - Outer test AUC
# - Aradaki fark (gap)
# tutulur.
#
# Gap = inner_mean - outer_test
# Büyük gap → overfitting sinyali

nested_scores = {
    'outer_fold': [],
    'class': [],
    'inner_cv_auc_mean': [],
    'inner_cv_auc_std': [],
    'outer_test_auc': [],
    'gap': []
}

# ---------------------------
# 2.3) Outer loop (asıl test simülasyonu)
# ---------------------------
for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_train, y_train)
):
    # Outer fold’un kendi train ve test’i
    X_tr_outer = X_train.iloc[train_idx]
    X_te_outer = X_train.iloc[test_idx]
    y_tr_outer = y_train.iloc[train_idx]
    y_te_outer = y_train.iloc[test_idx]

    # ---------------------------
    # 2.3.1) Inner loop sonuçları (sınıf bazlı)
    # ---------------------------
    # Her sınıf için AUC’leri ayrı listelerde topluyoruz
    inner_aucs_per_class = {c: [] for c in classes}

    for inner_train_idx, inner_val_idx in inner_cv.split(
        X_tr_outer, y_tr_outer
    ):
        X_tr_inner = X_tr_outer.iloc[inner_train_idx]
        X_val_inner = X_tr_outer.iloc[inner_val_idx]
        y_tr_inner = y_tr_outer.iloc[inner_train_idx]
        y_val_inner = y_tr_outer.iloc[inner_val_idx]

        # ÖNEMLİ:
        # - SMOTE YOK
        # - clone(model): her inner fold tamamen bağımsız
        est_inner = clone(model)
        est_inner.fit(X_tr_inner, y_tr_inner)

        # Validation set için olasılıklar
        y_proba_inner = est_inner.predict_proba(X_val_inner)

        # Multiclass → OVR binarization
        y_bin_inner = label_binarize(
            y_val_inner,
            classes=classes
        )

        # Her sınıf için AUC hesapla
        for i, c in enumerate(classes):
            try:
                auc_c = roc_auc_score(
                    y_bin_inner[:, i],
                    y_proba_inner[:, i]
                )
            except:
                # Tek sınıf düşerse AUC hesaplanamayabilir
                auc_c = 0.5

            inner_aucs_per_class[c].append(auc_c)

    # ---------------------------
    # 2.3.2) Outer test değerlendirmesi
    # ---------------------------
    # Bu kısım “gerçek dünya testi” gibi davranır
    est_outer = clone(model)
    est_outer.fit(X_tr_outer, y_tr_outer)

    y_proba_outer = est_outer.predict_proba(X_te_outer)

    y_bin_outer = label_binarize(
        y_te_outer,
        classes=classes
    )

    # ---------------------------
    # 2.3.3) Inner vs Outer karşılaştırması
    # ---------------------------
    for i, c in enumerate(classes):
        inner_mean = np.mean(inner_aucs_per_class[c])
        inner_std = np.std(inner_aucs_per_class[c])

        try:
            outer_auc = roc_auc_score(
                y_bin_outer[:, i],
                y_proba_outer[:, i]
            )
        except:
            outer_auc = 0.5

        # Gap:
        # Inner CV’de çok iyi → Outer test’te düşüyorsa
        # bu model sınıfa özel overfit olabilir
        gap = inner_mean - outer_auc

        nested_scores['outer_fold'].append(outer_fold)
        nested_scores['class'].append(f'RCB-{c}')
        nested_scores['inner_cv_auc_mean'].append(inner_mean)
        nested_scores['inner_cv_auc_std'].append(inner_std)
        nested_scores['outer_test_auc'].append(outer_auc)
        nested_scores['gap'].append(gap)

# ---------------------------
# 2.4) Nested CV sonuçlarının özeti
# ---------------------------
nested_df = pd.DataFrame(nested_scores)

print("\nNested CV Sonuçları (Sınıf Bazlı):")
print(
    nested_df.groupby('class').agg({
        'inner_cv_auc_mean': ['mean', 'std'],
        'outer_test_auc': ['mean', 'std'],
        'gap': ['mean', 'std']
    }).round(3)
)

# ---------------------------
# 2.5) Overfitting uyarı sistemi
# ---------------------------
# Klinik/akademik pratikte:
# - Gap > 0.15 → ciddi overfitting şüphesi
# - Gap 0.10–0.15 → dikkatli yorumlanmalı
# - Gap < 0.10 → genellikle kabul edilebilir

print("\n OVERFITTING UYARISI:")

for c in classes:
    gap_mean = nested_df[
        nested_df['class'] == f'RCB-{c}'
    ]['gap'].mean()

    if gap_mean > 0.15:
        print(
            f"  RCB-{c}: Gap = {gap_mean:.3f} "
            f"> 0.15 → OVERFITTING RİSKİ!"
        )
    elif gap_mean > 0.10:
        print(
            f"  RCB-{c}: Gap = {gap_mean:.3f} "
            f"> 0.10 → Dikkatli yorumlanmalı"
        )
    else:
        print(
            f"  RCB-{c}: Gap = {gap_mean:.3f} "
            f"→ Kabul edilebilir"
        )

# =============================================================================
# 3) SINIF BAZLI BOOTSTRAP CI'LARI
# =============================================================================
# Bu bölümün amacı:
# - Test setindeki performans metriklerinin
#   örnekleme belirsizliğini (sampling uncertainty) ölçmek.
#
# Neden bootstrap?
# - Tek bir test set skoru rastlantısal olabilir
# - Özellikle sınıf örnek sayısı azsa (örn. RCB-1)
# - Bootstrap ile %95 güven aralığı (CI) hesaplanır
#
# Klinik yorum:
# - Dar CI → güvenilir metrik
# - Geniş CI → dikkatli yorum
# - CI’lar sınıflar arasında karşılaştırma sağlar

print("\n=== SINIF BAZLI BOOTSTRAP CI'LARI ===")
print("Her RCB sınıfı için ayrı bootstrap CI (500 tekrar, %95)\n")

# ---------------------------
# 3.1) Bootstrap fonksiyonu (tek sınıf için)
# ---------------------------
def bootstrap_class_metrics(
    y_true,
    y_pred,
    y_proba,
    class_idx,
    n_bootstrap=500,
    random_state=42
):
    """
    Belirli bir RCB sınıfı için bootstrap ile:
    - Precision
    - Recall
    - F1
    - AUC
    - Specificity
    metriklerinin ortalama ve %95 CI değerlerini hesaplar.
    """

    # Tekrarlanabilirlik için RNG sabitle
    np.random.seed(random_state)

    # pandas Series → numpy array
    # (bootstrap indexleme sırasında uyumsuzluk olmasın diye)
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    n = len(y_true)

    # Metrikleri bootstrap boyunca depolayacağımız yapı
    metrics_list = {
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'specificity': []
    }

    # Multiclass → ilgili sınıf için binary dönüşüm
    y_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    proba_class = y_proba[:, class_idx]

    # ---------------------------
    # 3.1.1) Bootstrap döngüsü
    # ---------------------------
    for _ in range(n_bootstrap):

        # Test setinden, yerine koyarak örnekleme
        idx = np.random.choice(n, size=n, replace=True)

        y_true_boot = y_binary[idx]
        y_pred_boot = y_pred_binary[idx]
        proba_boot = proba_class[idx]

        # Confusion matrix elemanları
        tp = np.sum((y_true_boot == 1) & (y_pred_boot == 1))
        fp = np.sum((y_true_boot == 0) & (y_pred_boot == 1))
        fn = np.sum((y_true_boot == 1) & (y_pred_boot == 0))
        tn = np.sum((y_true_boot == 0) & (y_pred_boot == 0))

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall (Sensitivity)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1-score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        # Specificity (özellikle klinik için önemli)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC (binary OVR)
        try:
            auc_val = roc_auc_score(y_true_boot, proba_boot)
        except:
            # Tek sınıf düşerse AUC hesaplanamaz
            auc_val = 0.5

        # Değerleri listeye ekle
        metrics_list['precision'].append(precision)
        metrics_list['recall'].append(recall)
        metrics_list['f1'].append(f1)
        metrics_list['auc'].append(auc_val)
        metrics_list['specificity'].append(specificity)

    # ---------------------------
    # 3.2) %95 Confidence Interval hesaplama
    # ---------------------------
    results = {}

    for metric_name, values in metrics_list.items():
        values = np.array(values)

        results[metric_name] = {
            'mean': np.mean(values),
            'ci_lower': np.percentile(values, 2.5),
            'ci_upper': np.percentile(values, 97.5),
            'median': np.median(values)
        }

    return results

# ---------------------------
# 3.3) Test seti üzerinde bootstrap uygulama
# ---------------------------
class_bootstrap_results = {}

for c in classes:
    print(f"\nRCB-{c} için Bootstrap CI:")

    results = bootstrap_class_metrics(
        y_test,
        y_pred_test,
        y_proba_test,
        class_idx=c,
        n_bootstrap=500
    )

    class_bootstrap_results[f'RCB-{c}'] = results

    # Konsola okunabilir özet yazdır
    for metric_name, stats in results.items():
        print(
            f"  {metric_name.capitalize()}: "
            f"{stats['mean']:.3f} "
            f"(%95 CI: {stats['ci_lower']:.3f} - {stats['ci_upper']:.3f}, "
            f"medyan: {stats['median']:.3f})"
        )

# ---------------------------
# 3.4) Bootstrap CI sonuçlarını tabloya dök
# ---------------------------
bootstrap_table = []

for c in classes:
    for metric_name, stats in class_bootstrap_results[f'RCB-{c}'].items():
        bootstrap_table.append({
            'RCB_Sınıfı': f'RCB-{c}',
            'Metrik': metric_name,
            'Mean': stats['mean'],
            'CI_Lower': stats['ci_lower'],
            'CI_Upper': stats['ci_upper'],
            'Median': stats['median']
        })

bootstrap_df = pd.DataFrame(bootstrap_table)

print("\n=== BOOTSTRAP CI TABLOSU ===")
print(bootstrap_df.to_string(index=False))

# =============================================================================
# 4) DECISION CURVE ANALYSIS (DCA)
# =============================================================================
# Decision Curve Analysis (DCA) şunu sorar:
#
# “Bu modeli kullanmak, hiçbir şey yapmamaya veya herkesi tedavi etmeye
#  kıyasla klinik olarak gerçekten fayda sağlıyor mu?”
#
# DCA; accuracy, AUC gibi istatistiklerden farklı olarak:
# - Yanlış pozitiflerin klinik maliyetini
# - Yanlış negatiflerin klinik riskini
# eşik olasılık (threshold probability) üzerinden değerlendirir.
#
# Klinik anlam:
# - Model eğrisi, Treat-all ve Treat-none eğrilerinin ÜZERİNDEYSE
#   → Model klinik fayda sağlar
# - Altındaysa → Model kullanımı önerilmez

print("\n=== DECISION CURVE ANALYSIS ===")
print("Her RCB sınıfı için DCA eğrisi oluşturuluyor...\n")

# ---------------------------
# 4.1) Net Benefit hesaplama fonksiyonu
# ---------------------------
def calculate_net_benefit(y_true, y_proba, threshold):
    """
    Belirli bir eşik olasılık (threshold probability) için:
    modelin net klinik faydasını hesaplar.

    Net Benefit formülü:
    NB = (TP / N) − (FP / N) × (threshold / (1 − threshold))
    """

    n = len(y_true)

    # Olasılığa göre binary karar (treat / not treat)
    y_pred = (y_proba >= threshold).astype(int)

    # True Positive ve False Positive sayıları
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # Aşırı uçlarda matematiksel anlamsızlık önleme
    if threshold == 0 or threshold == 1:
        return 0.0

    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit

# ---------------------------
# 4.2) Treat-all stratejisi
# ---------------------------
def calculate_treat_all_net_benefit(y_true, threshold):
    """
    Treat-all stratejisi:
    - Herkese müdahale ediliyor varsayımı
    - Klinik karşılaştırma için referans çizgisi
    """

    n = len(y_true)

    tp = np.sum(y_true == 1)
    fp = n - tp

    if threshold == 0 or threshold == 1:
        return 0.0

    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit

# ---------------------------
# 4.3) DCA için eşik olasılık aralığı
# ---------------------------
# Klinik pratikte genellikle:
# 0.1 – 0.8 arası eşikler anlamlı kabul edilir
thresholds = np.arange(0.1, 0.81, 0.05)

# ---------------------------
# 4.4) DCA grafiği hazırlığı
# ---------------------------
fig_dca, ax_dca = plt.subplots(figsize=(12, 8))

colors = ['blue', 'orange', 'green', 'red']
class_names = ['RCB-0', 'RCB-1', 'RCB-2', 'RCB-3']

# ---------------------------
# 4.5) Her sınıf için DCA eğrisi
# ---------------------------
for class_idx, (c, color, name) in enumerate(zip(classes, colors, class_names)):

    # Multiclass → binary dönüşüm (one-vs-rest)
    y_true_binary = (y_test.values == c).astype(int)
    y_proba_binary = y_proba_test[:, class_idx]

    # Modelin net faydası
    net_benefits = []

    for pt in thresholds:
        nb = calculate_net_benefit(
            y_true_binary,
            y_proba_binary,
            pt
        )
        net_benefits.append(nb)

    # Treat-all eğrisi (sadece RCB-0 için çiziliyor → görsel karmaşayı azaltmak için)
    if class_idx == 0:
        treat_all_nb = [
            calculate_treat_all_net_benefit(y_true_binary, pt)
            for pt in thresholds
        ]

        ax_dca.plot(
            thresholds,
            treat_all_nb,
            '--',
            color='gray',
            linewidth=2,
            label='Treat-all (RCB-0 için)',
            alpha=0.7
        )

    # Modelin DCA eğrisi
    ax_dca.plot(
        thresholds,
        net_benefits,
        '-',
        color=color,
        linewidth=2,
        label=f'Model ({name})',
        marker='o',
        markersize=4
    )

# ---------------------------
# 4.6) Treat-none stratejisi
# ---------------------------
# Hiç kimseye müdahale edilmediği senaryo
ax_dca.axhline(
    y=0,
    color='black',
    linestyle='-',
    linewidth=2,
    label='Treat-none'
)

# ---------------------------
# 4.7) Grafik düzenlemeleri
# ---------------------------
ax_dca.set_xlabel(
    'Threshold Probability',
    fontsize=12,
    fontweight='bold'
)
ax_dca.set_ylabel(
    'Net Benefit',
    fontsize=12,
    fontweight='bold'
)

ax_dca.set_title(
    'Decision Curve Analysis - Model ALL (LightGBM + SMOTE YOK)',
    fontsize=14,
    fontweight='bold'
)

ax_dca.legend(loc='best', fontsize=10)
ax_dca.grid(True, alpha=0.3)

ax_dca.set_xlim([0.1, 0.8])
ax_dca.set_ylim([-0.3, 0.5])

plt.tight_layout()

# ---------------------------
# 4.8) Şekli kaydet
# ---------------------------
BASE_DIR = 'outputs'
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.savefig(
    os.path.join(FIG_DIR, 'decision_curve_analysis_model_all_lgbm.png'),
    dpi=300,
    bbox_inches='tight'
)

print(
    f" DCA şekli kaydedildi: "
    f"{os.path.join(FIG_DIR, 'decision_curve_analysis_model_all_lgbm.png')}"
)

plt.close()

# ============================================================================
# 5) CALIBRATION CURVE ANALYSIS
# ============================================================================
# Kalibrasyon analizi şu soruya cevap verir:
#
# “Modelin tahmin ettiği olasılıklar, gerçek gerçekleşme
#  oranlarını ne kadar doğru yansıtıyor?”
#
# Örnek:
# - Model %80 diyorsa → gerçekten vakaların %80’i mi pozitif?
#
# Bu analizde üç yaklaşım karşılaştırılır:
# 1) Raw (ham model çıktısı)
# 2) Platt Scaling (sigmoid / logistic calibration)
# 3) Isotonic Regression (non-parametric calibration)
#
# Klinik açıdan:
# - Daha iyi kalibrasyon → daha güvenilir risk tahmini
# - Brier Score ↓ → kalibrasyon ↑

print("\n=== CALIBRATION CURVE ANALYSIS ===")
print("Raw, Platt ve Isotonic kalibrasyon yöntemleri uygulanıyor...\n")

# -------------------------------------------------------------------------
# 5.1) Multiclass → Micro-average dönüşüm
# -------------------------------------------------------------------------
# Multiclass kalibrasyon zor olduğu için:
# - Tüm sınıflar OVR olarak birleştirilir
# - Micro-average yaklaşımı kullanılır
#
# Bu yaklaşım:
# - Tüm sınıflardaki olasılıkları tek bir havuzda değerlendirir
# - Genel olasılık güvenilirliğini ölçer

y_true_binary_all = label_binarize(
    y_test,
    classes=classes
)

# Modelin ham olasılık çıktıları
y_proba_all = y_proba_test

# Micro-average flatten işlemi
y_true_micro = y_true_binary_all.ravel()
y_proba_micro = y_proba_all.ravel()

# -------------------------------------------------------------------------
# 5.2) Raw (kalibrasyonsuz) model eğrisi
# -------------------------------------------------------------------------
fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(
    y_true_micro,
    y_proba_micro,
    n_bins=10,
    strategy='uniform'
)

# -------------------------------------------------------------------------
# 5.3) Platt Scaling (Sigmoid Calibration)
# -------------------------------------------------------------------------
# Logistic regression ile olasılıkları yeniden ölçekler
# Avantaj:
# - Overfitting riski düşük
# - Küçük veri setlerinde stabil
#
# Dezavantaj:
# - Lineer varsayım yapar

calibrated_platt = CalibratedClassifierCV(
    model,
    method='sigmoid',
    cv=3
)

calibrated_platt.fit(X_train, y_train)

y_proba_platt = calibrated_platt.predict_proba(X_test)
y_proba_micro_platt = y_proba_platt.ravel()

fraction_of_positives_platt, mean_predicted_value_platt = calibration_curve(
    y_true_micro,
    y_proba_micro_platt,
    n_bins=10,
    strategy='uniform'
)

# -------------------------------------------------------------------------
# 5.4) Isotonic Regression
# -------------------------------------------------------------------------
# Non-parametric (esnek) kalibrasyon
# Avantaj:
# - Karmaşık kalibrasyon hatalarını yakalar
#
# Dezavantaj:
# - Küçük veri setlerinde overfitting riski olabilir

calibrated_isotonic = CalibratedClassifierCV(
    model,
    method='isotonic',
    cv=3
)

calibrated_isotonic.fit(X_train, y_train)

y_proba_isotonic = calibrated_isotonic.predict_proba(X_test)
y_proba_micro_isotonic = y_proba_isotonic.ravel()

fraction_of_positives_isotonic, mean_predicted_value_isotonic = calibration_curve(
    y_true_micro,
    y_proba_micro_isotonic,
    n_bins=10,
    strategy='uniform'
)

# -------------------------------------------------------------------------
# 5.5) Brier Score hesaplama
# -------------------------------------------------------------------------
# Brier Score:
# - Olasılık tahmininin doğruluğunu ölçer
# - Daha küçük değer → daha iyi kalibrasyon

brier_raw = brier_score_loss(
    y_true_micro,
    y_proba_micro
)

brier_platt = brier_score_loss(
    y_true_micro,
    y_proba_micro_platt
)

brier_isotonic = brier_score_loss(
    y_true_micro,
    y_proba_micro_isotonic
)

print(f"Brier Score - Raw: {brier_raw:.3f}")
print(f"Brier Score - Platt: {brier_platt:.3f}")
print(f"Brier Score - Isotonic: {brier_isotonic:.3f}")
print("(Daha düşük Brier → Daha iyi kalibrasyon)")

# -------------------------------------------------------------------------
# 5.6) Calibration Curve grafiği
# -------------------------------------------------------------------------
fig_cal, ax_cal = plt.subplots(figsize=(10, 8))

# Mükemmel kalibrasyon referansı
ax_cal.plot(
    [0, 1],
    [0, 1],
    'k--',
    label='Perfect Calibration',
    linewidth=2
)

# Raw model
ax_cal.plot(
    mean_predicted_value_raw,
    fraction_of_positives_raw,
    'o-',
    color='blue',
    linewidth=2,
    markersize=8,
    label=f'Raw (Brier={brier_raw:.3f})'
)

# Platt scaling
ax_cal.plot(
    mean_predicted_value_platt,
    fraction_of_positives_platt,
    's--',
    color='orange',
    linewidth=2,
    markersize=6,
    label=f'Platt (Brier={brier_platt:.3f})'
)

# Isotonic regression
ax_cal.plot(
    mean_predicted_value_isotonic,
    fraction_of_positives_isotonic,
    'D-',
    color='green',
    linewidth=2,
    markersize=6,
    label=f'Isotonic (Brier={brier_isotonic:.3f})'
)

ax_cal.set_xlabel(
    'Tahmin edilen olasılık (bin ortalaması)',
    fontsize=12,
    fontweight='bold'
)
ax_cal.set_ylabel(
    'Ampirik olasılık',
    fontsize=12,
    fontweight='bold'
)

ax_cal.set_title(
    'Kalibrasyon Eğrileri - Model ALL (LightGBM + SMOTE YOK)\n'
    '(micro-average; daha düşük Brier → daha iyi)',
    fontsize=13,
    fontweight='bold'
)

ax_cal.legend(loc='best', fontsize=10)
ax_cal.grid(True, alpha=0.3)
ax_cal.set_xlim([0, 1])
ax_cal.set_ylim([0, 1])

plt.tight_layout()

plt.savefig(
    os.path.join(FIG_DIR, 'calibration_curve_model_all_lgbm.png'),
    dpi=300,
    bbox_inches='tight'
)

print(
    f" Calibration Curve şekli kaydedildi: "
    f"{os.path.join(FIG_DIR, 'calibration_curve_model_all_lgbm.png')}"
)

plt.close()


