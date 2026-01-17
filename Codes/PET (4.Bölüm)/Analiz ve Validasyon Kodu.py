# =============================================================================
# BÖLÜM 1 — IMPORTLAR + GENEL AYARLAR + VERİ YÜKLEME + HAZIRLIK
# =============================================================================
# Amaç:
# - Analizde kullanacağımız tüm kütüphaneleri import etmek
# - Veriyi Excel'den yüklemek (Colab upload)
# - PET feature’ları eksiksiz olan hastaları seçmek (kohort tanımı)
# - Hedef değişkeni (RCB_Kategorize) modelin anlayacağı şekilde sayısal encode etmek
# - Model varyantları için X setlerini (ALL / PET / ALL+PET) hazırlamak

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Modelleme algoritmaları
# -----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb

# -----------------------------------------------------------------------------
# Model değerlendirme / eğitim yardımcıları
# -----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
# train_test_split: train/test ayırmak için
# StratifiedKFold: sınıf oranlarını fold’larda korumak için
# cross_validate: CV skorlarını metrik bazında almak için

from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    f1_score, roc_curve, auc, classification_report,
    precision_recall_fscore_support, brier_score_loss
)
from sklearn.preprocessing import label_binarize, LabelEncoder

# -----------------------------------------------------------------------------
# Model açıklanabilirlik / önem analizi
# -----------------------------------------------------------------------------
from sklearn.inspection import permutation_importance
# permutation_importance:
# - bir özelliği karıştırıp (permute) performans değişimine bakar
# - "bu feature gerçekten faydalı mı?" sorusuna daha güvenilir yanıt verir
# - özellikle PET feature’larının katkısını raporlamak için çok iyi

# -----------------------------------------------------------------------------
# Kalibrasyon analizi
# -----------------------------------------------------------------------------
from sklearn.calibration import calibration_curve
# calibration_curve:
# - modelin ürettiği olasılıkların "gerçek olasılık" ile ne kadar uyumlu olduğuna bakar
# - klinik karar sistemleri için önemli olabilir

# -----------------------------------------------------------------------------
# Dengesiz sınıflar (imbalance) için SMOTE
# -----------------------------------------------------------------------------
from imblearn.over_sampling import SMOTE
# SMOTE:
# - azınlık sınıflar için sentetik örnek üretir
# - burada ileride "SMOTE ile / SMOTE'suz" kıyas yapılabilir

# -----------------------------------------------------------------------------
# İstatistiksel testler
# -----------------------------------------------------------------------------
from scipy.stats import wilcoxon
# wilcoxon:
# - iki yöntem/model arasında eşleştirilmiş ölçümlerde fark var mı test eder
# - burada bootstrap AUC dizileri karşılaştırmak için kullanılmış

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
# warnings kapatma:
# - notebook çıktısını temiz tutar
# - ama debug aşamasında istersek kaldırabiliriz

print("="*80)
print("PET VERİLERİ - EKSİK ANALİZLER VE İSTATİSTİKSEL TESTLER")
print("="*80)

# =============================================================================
# 1. VERİ YÜKLEME VE HAZIRLIK
# =============================================================================
# Burada Colab üzerinden Excel dosyası yükleniyor.
# Dosya adı otomatik olarak alınıp pandas ile okunuyor.

from google.colab import files

print("\n📁 Excel dosyanızı yükleyin:")
uploaded = files.upload()
# uploaded: dict benzeri bir yapı döner {dosya_adi: bytes}

file_name = list(uploaded.keys())[0]
# Kullanıcı tek dosya yüklediği varsayımıyla ilk dosyayı seçiyoruz.

data = pd.read_excel(file_name)
# Excel okuma: sütun adları ve içerikler dataframe'e gelir.

print(f" Veri yüklendi: {data.shape}")
# shape: (satır_sayısı, sütun_sayısı)

# -----------------------------------------------------------------------------
# PET özellikleri (14 adet)
# -----------------------------------------------------------------------------
# Bu liste PET'ten türetilmiş nicel özellikleri temsil ediyor.
# "pet_features" ile PET verisi eksik olan hastaları filtreleyeceğiz.

pet_features = [
    'SUVmax', 'SUVmean4', 'TLG', 'MTV',
    'Yüzey/Hacim Oranı4', 'Küresellik4', 'Asferisite4',
    'SUV Varyansı4', 'SUV Eğriliği4',
    'GLCM Entropi4', 'GLCM Kontrast4',
    'GLRLM Non-Uniformite4', 'NGTDM Coarseness4', 'GLSZM Entropi4'
]

# -----------------------------------------------------------------------------
# Tüm klinik/radyolojik özellikler (62 adet)
# -----------------------------------------------------------------------------
# ALL feature’lar: klinik + radyoloji + laboratuvar vb.
# Daha önceki bölümdeki "Model ALL" seti.

all_features = [
    'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i12',
    'i13', 'i14', 'i15', 'i46', 'i47',
    'i16', 'i17', 'i18', 'i19', 'i45',
    'i21', 'i22', 'i23', 'i24', 'i25', 'i26', 'i27', 'i28', 'i29', 'i30',
    'i31', 'i32', 'i33', 'i34', 'i35', 'i36', 'i37', 'i38', 'i39', 'i40',
    'i41', 'i42', 'i43', 'i44',
    'i48', 'i49', 'i50', 'i51', 'i52', 'i53', 'i54', 'i55', 'i56', 'i57',
    'i58', 'i59', 'i60', 'i61', 'i62', 'i63', 'i64'
]

# -----------------------------------------------------------------------------
# Hedef değişken
# -----------------------------------------------------------------------------
target = 'RCB_Kategorize'
print(" Veri tipleri kontrol ediliyor ve düzeltiliyor...")
for feat in pet_features + all_features:
    if feat in data.columns:
        # Virgüllü sayıları noktaya çevirip gerçek sayıya (float) dönüştürür
        data[feat] = pd.to_numeric(data[feat].astype(str).str.replace(',', '.'), errors='coerce')

# Önce sadece hedefi boş olanları çıkaralım
data = data.dropna(subset=[target])

# Geri kalan sayısal boşlukları sütun ortalamasıyla dolduralım (Çökme riskini önler)
for feat in pet_features + all_features:
    if feat in data.columns:
        data[feat] = data[feat].fillna(data[feat].mean())
data_pet = data.copy()
print(f"\n PET verisi olan hasta sayısı: {len(data_pet)}")

# -----------------------------------------------------------------------------
# Hedef değişkeni sayısal encode edelim (modelin anlayacağı format)
# -----------------------------------------------------------------------------
# LabelEncoder:
# - target değerleri string olsa bile (örn "RCB-0") 0..k-1 formatına çevirir.
# - Böylece tüm modeller aynı sınıf kodlamasıyla eğitilir.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data_pet['RCB_encoded'] = le.fit_transform(data_pet[target])

print(f" RCB sınıfları: {le.classes_}")
# le.classes_:
# - orijinal sınıf isimlerini gösterir
# - raporlamada / confusion matrix label'larında kullanmak için önemli

# -----------------------------------------------------------------------------
# Model varyantları için X setlerini hazırlayacağız (BÖLÜM 2'de split başlıyor)
# -----------------------------------------------------------------------------
# X_all: sadece klinik/radyolojik
# X_pet: sadece PET
# X_all_pet: birleşik
X_all = data_pet[all_features]
X_pet = data_pet[pet_features]
X_all_pet = data_pet[all_features + pet_features]

# y: encode edilmiş hedef
y = data_pet['RCB_encoded']

# =============================================================================
# BÖLÜM 2 — SINIF DAĞILIMI ANALİZİ (FULL / TRAIN / TEST) + GÖRSELLEŞTİRME
# =============================================================================
# Amaç:
# - RCB sınıflarının veri içinde ne kadar dengesiz olduğunu görmek (class imbalance)
# - Stratified split'in gerçekten sınıf oranlarını koruyup korumadığını doğrulamak
# - Bu bilgiyi tezde "veri seti tanımı" olarak raporlayabilmek
#
# Neden kritik?
# - İmbalance varsa Accuracy tek başına yanıltıcı olabilir.
# - SMOTE / class_weight gibi yöntemlerin gerekliliği burada anlaşılır.
# - Train ve Test dağılımı farklıysa test sonuçları güvenilirliğini kaybeder.
#   (Stratify bunu azaltır.)

print("\n" + "="*80)
print(" SINIF DAĞILIMI ANALİZİ")
print("="*80)

# -----------------------------------------------------------------------------
# 2.1) Train-test split (stratified)
# -----------------------------------------------------------------------------
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,               # burada sadece split için X_all kullanıyoruz
    y,                   # hedef: stratify buna göre yapılacak
    test_size=0.2,       # %20 test
    random_state=42,     # tekrar üretilebilirlik
    stratify=y           # sınıf oranlarını koru
)

# Diğer setleri aynı indekslerle eşleyelim (Parantez dışında olmalı)
X_train_all_pet = X_all_pet.loc[X_train_all.index]
X_test_all_pet = X_all_pet.loc[X_test_all.index]

# -----------------------------------------------------------------------------
# 2.2) FULL veri seti sınıf dağılımı (PET kohortu)
# -----------------------------------------------------------------------------
# Burada "data_pet" zaten PET verisi olan hastaları içeriyor.
# O yüzden bu dağılım, PET analizi yapılacak popülasyonun sınıf dağılımıdır.

print("\n TÜM VERİ SETİ (PET kohortu):")

class_dist_full = data_pet[target].value_counts().sort_index()
# value_counts(): her sınıfın frekansını verir
# sort_index(): sınıfları alfabetik/sayısal sıraya koyar (raporlama tutarlı olur)

for cls, count in class_dist_full.items():
    # cls: orijinal target sınıf etiketi (örn "0" veya "RCB-0")
    # count: hasta sayısı
    pct = count / len(data_pet) * 100
    print(f"  {cls}: {count} hasta ({pct:.1f}%)")

# -----------------------------------------------------------------------------
# 2.3) Train seti sınıf dağılımı (encoded)
# -----------------------------------------------------------------------------
# y_train şu anda sayısal (0..k-1) kodlu.
# Tez/rapor için tekrar orijinal sınıf adlarına çevirmek daha okunur.

print("\n TRAIN SETİ:")

train_dist = pd.Series(y_train).value_counts().sort_index()
# y_train pandas Series olabilir ama güvenli olması için Series() ile sarmaladık.
# sort_index(): 0,1,2,3 sırası ile gösterir.

for cls_encoded, count in train_dist.items():
    cls_name = le.inverse_transform([cls_encoded])[0]
    # inverse_transform: encoded sınıfı orijinal etikete çevirir
    pct = count / len(y_train) * 100
    print(f"  {cls_name}: {count} hasta ({pct:.1f}%)")

# -----------------------------------------------------------------------------
# 2.4) Test seti sınıf dağılımı (encoded)
# -----------------------------------------------------------------------------
print("\n TEST SETİ:")

test_dist = pd.Series(y_test).value_counts().sort_index()

for cls_encoded, count in test_dist.items():
    cls_name = le.inverse_transform([cls_encoded])[0]
    pct = count / len(y_test) * 100
    print(f"  {cls_name}: {count} hasta ({pct:.1f}%)")

# -----------------------------------------------------------------------------
# 2.5) Görselleştirme: Full / Train / Test sınıf dağılımı
# -----------------------------------------------------------------------------
# Neden görsel?
# - Tezde "Veri seti özellikleri" kısmında tek bakışta imbalance gösterir.
# - Stratified split'in etkisini görsel olarak doğrular.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# 1 satır, 3 sütun: Full, Train, Test yan yana

# Her panel için:
# - data_y: o panelde çizilecek hedef dizisi
# - title: panel başlığı
for ax, (data_y, title) in zip(
    axes,
    [
        (y, "Tüm Veri Seti"),
        (y_train, "Train Seti"),
        (y_test, "Test Seti"),
    ]
):
    # Sınıf sayıları
    counts = pd.Series(data_y).value_counts().sort_index()

    # Etiketleri encoded->orijinal çevir
    labels = [le.inverse_transform([i])[0] for i in counts.index]

    # Bar plot
    
    
    ax.bar(
        labels,
        counts.values,
        color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][: len(labels)]
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Hasta Sayısı")
    ax.set_xlabel("RCB Sınıfı")

    # Çubukların üstüne sayı yazdır (okunabilirlik)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

plt.tight_layout()
# tight_layout: subplotların birbirine girmesini önler

plt.savefig("sinif_dagilimi_analizi.png", dpi=300, bbox_inches="tight")
print("\n Grafik kaydedildi: sinif_dagilimi_analizi.png")

plt.show()
# notebook içinde göster


# =============================================================================
# BÖLÜM 3 — NESTED CROSS-VALIDATION (GERÇEK NESTED YAPI) + OVERFITTING KONTROLÜ
# =============================================================================
# Amaç:
# - Model performansının "gerçekten" genellenebilir olup olmadığını test etmek
# - Özellikle tek bir split veya klasik CV'de görülen şüpheli yüksek AUC'leri
#   (örn. RCB-1 için AUC=1.000 gibi) daha katı bir yapıda doğrulamak
#
# Neden Nested CV?
# - Klasik CV: model + hiperparametreler + değerlendirme aynı döngüde olabilir
#   → iyimser (optimistic) skor riski
# - Nested CV:
#   * Outer loop  → gerçek test (model hiç görmedi)
#   * Inner loop  → model seçimi / tuning / stabilite
# - Tez için gold standard değerlendirme yaklaşımı olarak düşündüm.

print("\n" + "="*80)
print(" NESTED CROSS-VALIDATION ANALİZİ")
print("="*80)

from sklearn.base import clone

def nested_cv_evaluation(
    X,
    y,
    model_template,
    model_name,
    n_outer=5,
    n_inner=3,
    random_state=42
):
    """
    GERÇEK nested CV uygular.
    
    Parametreler:
    - X, y            : Feature matrisi ve hedef
    - model_template : SADECE şablon (asla direkt fit edilmez!)
    - model_name     : Yazdırma / raporlama için isim
    - n_outer        : Outer CV fold sayısı
    - n_inner        : Inner CV fold sayısı
    
    Çıktı:
    - outer_scores: outer fold test performansları
    """

    # -------------------------------------------------------------------------
    # 3.1) Outer CV tanımı (gerçek test döngüsü)
    # -------------------------------------------------------------------------
    outer_cv = StratifiedKFold(
        n_splits=n_outer,
        shuffle=True,
        random_state=random_state
    )

    # Outer test sonuçlarını saklamak için
    outer_scores = {
        "test_auc": [],
        "test_accuracy": [],
        "test_f1_macro": []
    }

    print(f"\n🔍 {model_name} için Nested CV başlatılıyor...")
    print(f"Outer folds: {n_outer}, Inner folds: {n_inner}\n")

    # -------------------------------------------------------------------------
    # 3.2) Outer loop — gerçek test değerlendirmesi
    # -------------------------------------------------------------------------
    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        # Outer train/test ayrımı
        X_train_outer = X.iloc[train_idx]
        X_test_outer  = X.iloc[test_idx]
        y_train_outer = y.iloc[train_idx]
        y_test_outer  = y.iloc[test_idx]

        # ---------------------------------------------------------------------
        # 3.2.1) Inner CV — modelin "kararlılığını" görmek için
        # ---------------------------------------------------------------------
        # Not:
        # - Bu örnekte hiperparametre araması yapılmıyor
        # - Ama inner CV hâlâ önemli:
        #   * fold içinde performansın ne kadar dalgalandığını gösterir
        #   * aşırı uyum (overfitting) sinyali verir
        inner_cv = StratifiedKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=random_state
        )

        inner_auc_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(
            X_train_outer, y_train_outer
        ):
            X_train_inner = X_train_outer.iloc[inner_train_idx]
            X_val_inner   = X_train_outer.iloc[inner_val_idx]
            y_train_inner = y_train_outer.iloc[inner_train_idx]
            y_val_inner   = y_train_outer.iloc[inner_val_idx]

            #  KRİTİK:
            # Her inner fold için YENİ model instance (clone)
            # Aksi halde önceki fit'lerden bilgi sızar (state leakage)
            model_inner = clone(model_template)

            model_inner.fit(X_train_inner, y_train_inner)

            y_val_proba = model_inner.predict_proba(X_val_inner)

            # Multiclass AUC (OVR + macro)
            y_val_bin = label_binarize(
                y_val_inner,
                classes=np.unique(y)
            )

            try:
                auc_inner = roc_auc_score(
                    y_val_bin,
                    y_val_proba,
                    multi_class="ovr",
                    average="macro"
                )
            except Exception:
                # Çok küçük fold'larda tek sınıf kalabilir
                auc_inner = 0.5

            inner_auc_scores.append(auc_inner)

        inner_auc_mean = np.mean(inner_auc_scores)
        inner_auc_std  = np.std(inner_auc_scores)

        # ---------------------------------------------------------------------
        # 3.2.2) Outer test — model ilk kez bu veriyi görüyor
        # ---------------------------------------------------------------------
        model_outer = clone(model_template)
        model_outer.fit(X_train_outer, y_train_outer)

        y_test_pred  = model_outer.predict(X_test_outer)
        y_test_proba = model_outer.predict_proba(X_test_outer)

        y_test_bin = label_binarize(
            y_test_outer,
            classes=np.unique(y)
        )

        try:
            auc_outer = roc_auc_score(
                y_test_bin,
                y_test_proba,
                multi_class="ovr",
                average="macro"
            )
        except Exception:
            auc_outer = 0.5

        acc_outer = accuracy_score(y_test_outer, y_test_pred)
        f1_outer  = f1_score(y_test_outer, y_test_pred, average="macro")

        outer_scores["test_auc"].append(auc_outer)
        outer_scores["test_accuracy"].append(acc_outer)
        outer_scores["test_f1_macro"].append(f1_outer)

        # ---------------------------------------------------------------------
        # 3.2.3) Overfitting farkı (inner vs outer)
        # ---------------------------------------------------------------------
        gap = inner_auc_mean - auc_outer

        print(
            f"Fold {outer_fold}: "
            f"Inner AUC={inner_auc_mean:.3f}±{inner_auc_std:.3f} | "
            f"Outer AUC={auc_outer:.3f} | "
            f"Gap={gap:.3f}"
        )

    # -------------------------------------------------------------------------
    # 3.3) Nested CV özet raporu
    # -------------------------------------------------------------------------
    print(f"\n {model_name} — Nested CV Özet:")

    for metric, values in outer_scores.items():
        mean_val = np.mean(values)
        std_val  = np.std(values)
        print(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")

    # Overfitting yorumu
    mean_gap = (
        np.mean(inner_auc_scores) - np.mean(outer_scores["test_auc"])
        if len(outer_scores["test_auc"]) > 0
        else 0
    )

    print("\n Overfitting Yorumu:")
    if mean_gap > 0.15:
        print("  Ciddi overfitting şüphesi (gap > 0.15)")
    elif mean_gap > 0.10:
        print("  Orta düzey overfitting ihtimali (0.10 < gap ≤ 0.15)")
    else:
        print("  Kabul edilebilir genelleme farkı")

    return outer_scores

# =============================================================================
# 3.4) Nested CV UYGULAMALARI
# =============================================================================
# Fit işlemleri fonksiyon içinde clone() ile yapılıyor.

# Model ALL — RandomForest
rf_template = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

nested_scores_all = nested_cv_evaluation(
    X_all,
    y,
    rf_template,
    model_name="Model ALL (RandomForest)"
)

# Model ALL+PET — LightGBM
lgb_template = lgb.LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1)
nested_scores_all_pet = nested_cv_evaluation(
    X_all_pet,
    y,
    lgb_template,
    model_name="Model ALL+PET (LightGBM)"
)

# =============================================================================
# BÖLÜM 4 — BOOTSTRAP AUC GÜVEN ARALIKLARI + İSTATİSTİKSEL KARŞILAŞTIRMA
# =============================================================================
# Amaç:
# - Tek bir test AUC değerine körü körüne güvenmemek
# - Model performansının belirsizliğini (uncertainty) nicel olarak göstermek
# - Model ALL vs Model ALL+PET farkının:
#     * istatistiksel olarak anlamlı mı?
#     * yoksa örnekleme rastlantısından mı kaynaklı?
#   olduğunu test etmek
#
# Bu bölüm TEZ için kritik:
# - "AUC = 0.82" tek başına zayıftır
# - "AUC = 0.82 (95% CI: 0.76–0.88)" bilimsel olarak güçlüdür

print("\n" + "="*80)
print(" BOOTSTRAP AUC + İSTATİSTİKSEL KARŞILAŞTIRMA")
print("="*80)

from scipy.stats import wilcoxon

# =============================================================================
# 4.1) Bootstrap AUC fonksiyonu
# =============================================================================
def bootstrap_auc(
    X_test,
    y_test,
    model_fitted,
    n_iterations=500,
    random_state=42
):
    """
    Test seti üzerinde bootstrap ile AUC dağılımı üretir.

    Neden bootstrap?
    - Test seti tek bir örneklemdir
    - Bootstrap, test setini tekrar tekrar örnekleyerek
      AUC'nin dağılımını yaklaşık olarak çıkarır

    Parametreler:
    - X_test        : Test feature matrisi
    - y_test        : Test etiketleri (encoded)
    - model_fitted  : DAHA ÖNCE FIT EDİLMİŞ model (pipeline veya classifier)
    - n_iterations  : Bootstrap tekrar sayısı (≥500 önerilir)

    Returns:
    - auc_scores    : bootstrap AUC değerleri (numpy array)
    """

    rng = np.random.RandomState(random_state)
    auc_scores = []

    # y_test pandas Series ise numpy array'e dönüştür
    y_true = y_test.values if hasattr(y_test, "values") else y_test

    # Multiclass AUC için binarize edilmiş hedef
    classes_sorted = np.unique(y_true)

    for i in range(n_iterations):
        # -------------------------------------------------------------
        # 4.1.1) Bootstrap örnekleme (test setinden, replacement ile)
        # -------------------------------------------------------------
        idx = rng.choice(len(X_test), size=len(X_test), replace=True)

        X_boot = X_test.iloc[idx]
        y_boot = y_true[idx]

        # -------------------------------------------------------------
        # 4.1.2) Olasılık tahminleri
        # -------------------------------------------------------------
        #  Model yeniden eğitilmez!
        # - Amaç: model belirsizliğini değil
        # - test örneklemesinin belirsizliğini ölçmek
        y_proba = model_fitted.predict_proba(X_boot)

        # -------------------------------------------------------------
        # 4.1.3) AUC hesaplama (macro OVR)
        # -------------------------------------------------------------
        y_bin = label_binarize(y_boot, classes=classes_sorted)

        try:
            auc_val = roc_auc_score(
                y_bin,
                y_proba,
                multi_class="ovr",
                average="macro"
            )
            auc_scores.append(auc_val)
        except Exception:
            # Çok küçük bootstrap örneklerinde tek sınıf kalabilir
            continue

    return np.array(auc_scores)

# =============================================================================
# 4.2) Final modellerin hazırlanması (TEST SET İÇİN)
# =============================================================================
# Burada artık CV yok.
# Amaç:
# - Final modelleri train setinin TAMAMI ile eğitmek
# - Test setinde bootstrap yapmak

# -----------------------------
# Model ALL (RandomForest)
# -----------------------------
model_all_final = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model_all_final.fit(X_train_all, y_train)

# -----------------------------
# Model ALL+PET (LightGBM)
# -----------------------------
model_all_pet_final = lgb.LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1)

model_all_pet_final.fit(X_train_all_pet, y_train)

# =============================================================================
# 4.3) Bootstrap AUC hesaplamaları
# =============================================================================
print("\n🔄 Bootstrap AUC hesaplanıyor...")

auc_boot_all = bootstrap_auc(
    X_test_all,
    y_test,
    model_all_final,
    n_iterations=500
)

auc_boot_all_pet = bootstrap_auc(
    X_test_all_pet,
    y_test,
    model_all_pet_final,
    n_iterations=500
)

# =============================================================================
# 4.4) Bootstrap özet istatistikleri
# =============================================================================
def summarize_bootstrap(name, auc_scores):
    mean = np.mean(auc_scores)
    std  = np.std(auc_scores)
    ci_l = np.percentile(auc_scores, 2.5)
    ci_u = np.percentile(auc_scores, 97.5)

    print(f"\n{name}")
    print(f"  Mean AUC : {mean:.3f}")
    print(f"  Std      : {std:.3f}")
    print(f"  95% CI   : [{ci_l:.3f}, {ci_u:.3f}]")

    return mean, ci_l, ci_u

mean_all, ci_l_all, ci_u_all = summarize_bootstrap(
    "Model ALL (RF)", auc_boot_all
)

mean_all_pet, ci_l_all_pet, ci_u_all_pet = summarize_bootstrap(
    "Model ALL+PET (LGBM)", auc_boot_all_pet
)

# =============================================================================
# 4.5) Wilcoxon Signed-Rank Test
# =============================================================================
# Neden Wilcoxon?
# - Bootstrap AUC dağılımları:
#   * Normal dağılmak zorunda değil
#   * Aynı test setinden türetilmiş → bağımlı örnekler
# - Wilcoxon = non-parametrik + paired test → doğru seçim

stat, p_value = wilcoxon(auc_boot_all, auc_boot_all_pet)

print("\n Wilcoxon Signed-Rank Test")
print(f"  Test istatistiği: {stat:.3f}")
print(f"  p-değeri        : {p_value:.4f}")

if p_value < 0.05:
    print("  İstatistiksel olarak ANLAMLI fark (p < 0.05)")
else:
    print("  İstatistiksel olarak ANLAMLI fark yok (p ≥ 0.05)")

# =============================================================================
# 4.6) Bootstrap AUC dağılımlarının görselleştirilmesi
# =============================================================================
plt.figure(figsize=(10, 6))

plt.hist(
    auc_boot_all,
    bins=30,
    alpha=0.5,
    label="Model ALL",
    color="blue"
)

plt.hist(
    auc_boot_all_pet,
    bins=30,
    alpha=0.5,
    label="Model ALL+PET",
    color="red"
)

plt.axvline(mean_all, color="blue", linestyle="--", linewidth=2)
plt.axvline(mean_all_pet, color="red", linestyle="--", linewidth=2)

plt.xlabel("Bootstrap AUC")
plt.ylabel("Frekans")
plt.title(
    f"Bootstrap AUC Dağılımları\n"
    f"Wilcoxon p = {p_value:.4f}",
    fontsize=14,
    fontweight="bold"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bootstrap_auc_dagilimi.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n Bootstrap AUC analizi tamamlandı.")

# =============================================================================
# BÖLÜM 5 — SINIF BAZINDA DETAYLI METRİKLER (Klinik Odaklı Analiz)
# =============================================================================
# Amaç:
# - Genel (macro) skorların arkasındaki gerçeği görmek
# - Hangi RCB sınıflarının iyi / kötü öğrenildiğini açıkça ortaya koymak
# - Klinik açıdan kritik sınıflar (örn. RCB-1 vs RCB-2) için:
#     * kaç hasta kaçırılıyor?
#     * yanlış pozitif oranı ne?
#     * model hangi sınıfta güvenilir?
#
# Neden bu bölüm şart?
# - "AUC=0.85" tek başına klinikte anlamsızdır
# - Klinik karar sınıf bazlı verilir
# - Özellikle azınlık sınıflar macro AUC içinde gizlenebilir

print("\n" + "="*80)
print(" SINIF BAZINDA DETAYLI METRİKLER")
print("="*80)

from sklearn.metrics import classification_report

# =============================================================================
# 5.1) Sınıf bazlı metrikleri hesaplayan yardımcı fonksiyon
# =============================================================================
def class_wise_metrics(
    y_true,
    y_pred,
    y_proba,
    class_labels,
    class_names,
    model_name
):
    """
    Her sınıf için:
    - Precision
    - Recall (Sensitivity)
    - F1-score
    - AUC (One-vs-Rest)
    - Specificity

    hesaplar ve tablo olarak döndürür.
    """

    metrics = []

    for i, cls in enumerate(class_labels):
        # -------------------------------------------------------------
        # 5.1.1) Binary problem: ilgili sınıf vs geri kalanlar
        # -------------------------------------------------------------
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)

        # -------------------------------------------------------------
        # 5.1.2) Confusion matrix bileşenleri
        # -------------------------------------------------------------
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # -------------------------------------------------------------
        # 5.1.3) Sınıf bazlı AUC (OVR)
        # -------------------------------------------------------------
        try:
            auc_cls = roc_auc_score(y_true_bin, y_proba[:, i])
        except Exception:
            auc_cls = 0.5

        metrics.append({
            "Model": model_name,
            "Class": class_names[i],
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Specificity": specificity,
            "AUC": auc_cls,
            "Support": np.sum(y_true_bin)
        })

    return pd.DataFrame(metrics)

# =============================================================================
# 5.2) Final modellerden test tahminleri
# =============================================================================
# Bu bölümde:
# - BÖLÜM 4'te eğitilmiş final modeller kullanılır
# - Tekrar fit yapılmaz (data leakage yok)

# -----------------------------
# Model ALL
# -----------------------------
y_pred_all = model_all_final.predict(X_test_all)
y_proba_all = model_all_final.predict_proba(X_test_all)

# -----------------------------
# Model ALL+PET
# -----------------------------
y_pred_all_pet = model_all_pet_final.predict(X_test_all_pet)
y_proba_all_pet = model_all_pet_final.predict_proba(X_test_all_pet)

class_labels = np.unique(y_test)
class_names  = le.inverse_transform(class_labels)

# =============================================================================
# 5.3) Sınıf bazlı metrik tabloları
# =============================================================================
metrics_all = class_wise_metrics(
    y_test.values,
    y_pred_all,
    y_proba_all,
    class_labels,
    class_names,
    model_name="Model ALL"
)

metrics_all_pet = class_wise_metrics(
    y_test.values,
    y_pred_all_pet,
    y_proba_all_pet,
    class_labels,
    class_names,
    model_name="Model ALL+PET"
)

print("\n Model ALL — Sınıf Bazlı Metrikler")
print(metrics_all.round(3).to_string(index=False))

print("\n Model ALL+PET — Sınıf Bazlı Metrikler")
print(metrics_all_pet.round(3).to_string(index=False))

# =============================================================================
# 5.4) Klinik yorum için karşılaştırmalı tablo
# =============================================================================
comparison = metrics_all.merge(
    metrics_all_pet,
    on="Class",
    suffixes=("_ALL", "_ALL_PET")
)

comparison["ΔRecall"] = (
    comparison["Recall_ALL_PET"] - comparison["Recall_ALL"]
)

comparison["ΔAUC"] = (
    comparison["AUC_ALL_PET"] - comparison["AUC_ALL"]
)

print("\n SINIF BAZLI KARŞILAŞTIRMA (ALL vs ALL+PET)")
print(comparison.round(3).to_string(index=False))

# =============================================================================
# 5.5) Klinik yorum rehberi (tez metnine birebir girebilir)
# =============================================================================
print("\n KLİNİK YORUM REHBERİ")
print("-" * 60)

for _, row in comparison.iterrows():
    cls = row["Class"]

    if row["ΔRecall"] > 0.05:
        comment = "PET eklenmesi duyarlılığı anlamlı artırmıştır."
    elif row["ΔRecall"] < -0.05:
        comment = "PET eklenmesi duyarlılığı düşürmüştür."
    else:
        comment = "PET eklenmesi duyarlılığı belirgin değiştirmemiştir."

    print(f"{cls}: {comment}")

# =============================================================================
# 5.6) Kaydetme (tez için tablo)
# =============================================================================
metrics_all.to_csv("class_metrics_model_all.csv", index=False)
metrics_all_pet.to_csv("class_metrics_model_all_pet.csv", index=False)
comparison.to_csv("class_metrics_comparison.csv", index=False)

print("\n Sınıf bazlı metrik tabloları kaydedildi.")

# =============================================================================
# BÖLÜM 6 — PERMUTATION IMPORTANCE (PET ÖZELLİKLERİNİN GERÇEK KATKISI)
# =============================================================================
# Amaç:
# - Modelin “hangi özellikleri gerçekten kullandığını” ölçmek
# - Tree-based modellerde görülen:
#     * Gini / gain importance yanlılıklarını (bias)
#     * korelasyon nedeniyle şişmiş önem skorlarını
#   bertaraf etmek
#
# Neden Permutation Importance?
# - Model-agnostic (RF, LGBM, XGB hepsi için geçerli)
# - Test seti üzerinde hesaplanır → genelleme odaklı
# - “Bu özelliği bozarsam performans ne kadar düşüyor?” sorusuna cevap verir
#
# Klinik yorum:
# - Pozitif importance → model bu özelliğe gerçekten ihtiyaç duyuyor
# - 0 civarı → redundant / bilgi taşımıyor
# - Negatif → gürültü / modele zarar veriyor olabilir

print("\n" + "="*80)
print(" PERMUTATION IMPORTANCE ANALİZİ")
print("="*80)

from sklearn.inspection import permutation_importance

# =============================================================================
# 6.1) Permutation importance hesaplama
# =============================================================================
# Burada özellikle:
# - Model ALL+PET
# - Final, test seti performansı raporlanan model
# kullanılıyor.
#
#  Çok önemli:
# - CV veya train seti DEĞİL
# - SADECE test seti
# Çünkü amaç:
# - modelin “genelleme” aşamasında hangi özelliklere dayandığını görmek

print("\n Model ALL+PET için permutation importance hesaplanıyor...")

perm_result = permutation_importance(
    model_all_pet_final,      # daha önce fit edilmiş final model
    X_test_all_pet,           # test seti
    y_test,                   # test etiketleri
    n_repeats=5,             # her özellik için 5 permütasyon
    random_state=42,
    scoring="roc_auc_ovr"     # multiclass AUC (macro)
)

# perm_result.importances_mean:
# - her feature için ortalama performans düşüşü
# - >0: faydalı, <0: zararlı

# =============================================================================
# 6.2) PET özelliklerini ayırma
# =============================================================================
# X_test_all_pet kolon sırası:
# [ALL özellikleri | PET özellikleri]
#
# Bu yüzden PET özelliklerinin indeksleri:
pet_start_idx = len(all_features)
pet_feature_indices = list(
    range(pet_start_idx, pet_start_idx + len(pet_features))
)

pet_perm_importance = {
    feat: perm_result.importances_mean[idx]
    for feat, idx in zip(pet_features, pet_feature_indices)
}

# Sırala (en faydalıdan en zararlıya)
pet_perm_importance_sorted = dict(
    sorted(
        pet_perm_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
)

print("\n PET ÖZELLİKLERİ — PERMUTATION IMPORTANCE:")
for feat, imp in pet_perm_importance_sorted.items():
    sign = "⬆️" if imp > 0 else "⬇️"
    print(f"  {feat:25s}: {imp:+.4f} {sign}")

# =============================================================================
# 6.3) Görselleştirme
# =============================================================================
# Klinik sunumlar ve tez için:
# - Hangi PET feature gerçekten katkı sağlıyor?
# - Hangileri zararlı / anlamsız?

features = list(pet_perm_importance_sorted.keys())
importances = list(pet_perm_importance_sorted.values())

colors = ["green" if imp > 0 else "red" for imp in importances]

plt.figure(figsize=(10, 8))
plt.barh(features, importances, color=colors, alpha=0.75)

plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Permutation Importance (Δ AUC)", fontsize=12)
plt.title(
    "PET Özellikleri — Permutation Importance\n"
    "(Pozitif: Faydalı | Negatif: Zararlı)",
    fontsize=14,
    fontweight="bold"
)

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("pet_permutation_importance.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n PET permutation importance grafiği kaydedildi.")

# =============================================================================
# 6.4) Klinik yorum rehberi (tez metnine birebir uyumlu)
# =============================================================================
print("\n KLİNİK YORUM REHBERİ")
print("-" * 60)

for feat, imp in pet_perm_importance_sorted.items():
    if imp > 0.01:
        comment = "model performansına anlamlı katkı sağlamaktadır."
    elif imp > 0:
        comment = "sınırlı katkı sağlamaktadır."
    else:
        comment = "model performansını düşürmektedir veya gürültü içermektedir."

    print(f"{feat}: {comment}")

# =============================================================================
# 6.5) Kaydetme 
# =============================================================================
perm_df = pd.DataFrame({
    "Feature": pet_perm_importance_sorted.keys(),
    "Permutation_Importance": pet_perm_importance_sorted.values()
})

perm_df.to_csv("pet_permutation_importance.csv", index=False)

print("\n✅ Permutation importance tablosu kaydedildi.")

# =============================================================================
# BÖLÜM 7 — CALIBRATION CURVES (OLASILIK GÜVENİLİRLİĞİ ve KLİNİK KARAR AÇISINDAN)
# =============================================================================
# Amaç:
# - Modelin verdiği olasılıkların "güvenilir" olup olmadığını değerlendirmek
# - Yani:
#     * Model %70 diyorsa → gerçekten %70 hasta mı pozitif?
# - Bu analiz özellikle klinik karar eşikleri (threshold) için kritiktir
#
# Neden calibration önemli?
# - AUC sadece sıralama gücünü ölçer
# - Klinik kararlar olasılığa dayanır:
#     * Tedavi başla / başlama
#     * Daha agresif yaklaş / bekle
# - Kötü kalibre bir model yüksek AUC’ye rağmen klinikte zararlı olabilir

print("\n" + "="*80)
print(" CALIBRATION CURVE ANALİZİ (Sınıf Bazında)")
print("="*80)

from sklearn.calibration import calibration_curve

# =============================================================================
# 7.1) Calibration curve hesaplayan yardımcı fonksiyon
# =============================================================================
def plot_calibration_curve(
    y_true,
    y_proba,
    class_idx,
    class_name,
    model_label,
    ax,
    n_bins=5
):
    """
    Tek bir sınıf için calibration curve çizer.

    Parametreler:
    - y_true      : gerçek etiketler (encoded)
    - y_proba     : predict_proba çıktısı
    - class_idx   : ilgili sınıfın indeksi
    - class_name  : görsel için sınıf adı
    - model_label : legend etiketi
    - ax          : matplotlib axis
    """

    # -------------------------------------------------------------
    # Binary problem: ilgili sınıf vs diğerleri
    # -------------------------------------------------------------
    y_true_bin = (y_true == class_idx).astype(int)
    y_proba_cls = y_proba[:, class_idx]

    # -------------------------------------------------------------
    # calibration_curve:
    # - predicted probability'leri bin'lere ayırır
    # - her bin'de gerçek pozitif oranını hesaplar
    # -------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(
        y_true_bin,
        y_proba_cls,
        n_bins=n_bins,
        strategy="quantile"
    )

    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=2,
        label=model_label
    )

    return prob_true, prob_pred

# =============================================================================
# 7.2) Sınıf bazında calibration curve'ler
# =============================================================================
# Burada iki model karşılaştırılıyor:
# - Model ALL
# - Model ALL+PET
#
# Her RCB sınıfı için ayrı grafik çizilir.
# Bu sayede:
# - Hangi sınıfta model güvenilir?
# - Hangi sınıfta aşırı / yetersiz güven var?
# net şekilde görülür.

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, cls in enumerate(class_labels):
    ax = axes[i]

    # Perfect calibration çizgisi
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")

    # Model ALL
    plot_calibration_curve(
        y_true=y_test.values,
        y_proba=y_proba_all,
        class_idx=cls,
        class_name=class_names[i],
        model_label="Model ALL",
        ax=ax
    )

    # Model ALL+PET
    plot_calibration_curve(
        y_true=y_test.values,
        y_proba=y_proba_all_pet,
        class_idx=cls,
        class_name=class_names[i],
        model_label="Model ALL+PET",
        ax=ax
    )

    ax.set_title(f"{class_names[i]} Calibration Curve", fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("calibration_curves.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n Calibration curve grafikleri kaydedildi.")

# =============================================================================
# 7.3) Klinik yorum rehberi (tez metni için)
# =============================================================================
print("\n KLİNİK YORUM REHBERİ")
print("-" * 60)

print("""
Calibration eğrileri şu şekilde yorumlanır:

1) Eğri diagonal çizgiye yakınsa:
   → Modelin olasılık tahminleri güvenilirdir.

2) Eğri diagonalin ÜZERİNDE ise:
   → Model gerçekte olandan DAHA DÜŞÜK olasılık tahmin ediyor
   → Under-confident model (ihtiyatlı)

3) Eğri diagonalin ALTINDA ise:
   → Model gerçekte olandan DAHA YÜKSEK olasılık tahmin ediyor
   → Over-confident model (klinik risk!)

Klinik açıdan:
- Over-confident model → gereksiz agresif tedavi riski
- Under-confident model → tedavide gecikme riski

Bu yüzden:
- AUC yüksek olsa bile
- calibration bozuksa
model klinik olarak güvenli değildir.
""")

# =============================================================================
# 7.4) Brier Score ile kalibrasyonun sayısal özeti
# =============================================================================
from sklearn.metrics import brier_score_loss

print("\n📏 BRIER SCORE (micro-average)")

# Micro-average yaklaşımı:
# - Tüm sınıfları tek binary problem gibi ele alır
y_true_bin_all = label_binarize(y_test.values, classes=class_labels).ravel()

# Brier Score için olasılıkları ve gerçek değerleri hizalıyoruz
brier_all = brier_score_loss(
    y_true_bin_all,
    y_proba_all.ravel()
)

brier_all_pet = brier_score_loss(
    y_true_bin_all,
    y_proba_all_pet.ravel()
)
print(f"Model ALL     Brier Score: {brier_all:.4f}")
print(f"Model ALL+PET Brier Score: {brier_all_pet:.4f}")

if brier_all_pet < brier_all:
    print("→ PET eklenmesi olasılık kalibrasyonunu İYİLEŞTİRMİŞTİR.")
else:
    print("→ PET eklenmesi olasılık kalibrasyonunu İYİLEŞTİRMEMİŞTİR.")

print("\n BÖLÜM 7 tamamlandı.")



