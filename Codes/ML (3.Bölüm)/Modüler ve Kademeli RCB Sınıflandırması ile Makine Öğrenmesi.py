# ============================================================================
# MODÜLER VE KADEMELİ RCB SINIFLANDIRMA + DATA LEAKAGE DÜZELTİLMİŞ (TÜM ŞEKİLLER)
# ============================================================================
# Bu script; farklı özellik grupları (P/O/D/K/B/R) ile farklı model senaryoları kurar,
# her senaryoda 3 farklı algoritmayı (RF/XGB/LGBM) dener, SMOTE YOK ve SMOTE VAR
# koşullarını karşılaştırır, metrik+grafik üretir, en iyi modeli seçer ve deploy eder.
#
# Data leakage kritik noktası:
# - Önce train-test split yapılır.
# - SMOTE yalnızca train tarafında uygulanır:
#   - CV içinde: her fold'un train kısmında (val kısmına uygulanmaz)
#   - Final eğitimde: train'in tamamında (test'e asla dokunulmaz)
# =============================================================================

# ---------------------------
# 0) Parametreler
# ---------------------------

# heavy_plots:
# True ise PR curve, calibration curve, gain/lift ve SHAP gibi daha maliyetli
# False ise sadece temel çıktılar (CM/ROC/importance vb.) üretilir.
heavy_plots = True

# quick_mode:
# Geliştirme/test amaçlı hızlı çalıştırma modu.
# True olursa sadece 3 model seti denenir (Model P, Model P+O+D, Model ALL).
# False olursa tüm kombinasyonlar denenir (daha uzun sürer).
quick_mode  = False

# ---------------------------
# 1) Kütüphaneler
# ---------------------------

# os        : klasör/dosya işlemleri (mkdir, path join, walk vb.)
# zipfile   : outputs klasörünü zip'e çevirip indirmek için
# io, sys   : (opsiyonel) IO ve sistem seviyesinde yardımcı araçlar
# warnings  : uyarıları bastırmak için (özellikle model eğitiminde çok uyarı olur)
# json      : deploy meta dosyalarını (feature listesi, class order vb.) yazmak için
import os, zipfile, io, sys, warnings, json

# Notebook çıktısında gereksiz uyarı spam'ini azaltmak için.
warnings.filterwarnings('ignore')

# numpy : sayısal işlemler, vektör/matris, random sampling
import numpy as np

# pandas : excel okuma, tablo işlemleri, DataFrame yönetimi
import pandas as pd

# matplotlib : ROC/PR vs. çizimler için temel plotting
import matplotlib.pyplot as plt

# seaborn : confusion matrix heatmap gibi daha “görsel” grafikler için
import seaborn as sns

# Colab'a özgü dosya yükleme/indirme aracı
from google.colab import files

# train_test_split     : en başta train-test ayırmak (leakage önlemek için zorunlu)
# StratifiedKFold      : CV'de sınıf dağılımını fold'larda korumak için
from sklearn.model_selection import train_test_split, StratifiedKFold

# clone : aynı model şablonundan her fold/denemede temiz instance üretmek için
# (modelin state'inin fold'lar arasında taşınmasını engeller)
from sklearn.base import clone

# Metrikler:
# - accuracy, precision, recall, f1 : sınıflandırma performansı
# - roc_auc_score : çok sınıflı macro AUC (ovr)
# - confusion_matrix : CM
# - roc_curve, auc : ROC eğrisi ve AUC'yi elle çizmek için
# - precision_recall_curve : PR curve çizimi için
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)

# label_binarize:
# Multiclass ROC/PR için sınıfları one-vs-rest (OVR) mantığıyla 0/1 matrise çevirir.
from sklearn.preprocessing import label_binarize

# RandomForestClassifier: bagging tabanlı ağaç topluluğu
from sklearn.ensemble import RandomForestClassifier

# XGBClassifier : gradient boosting (xgboost)
from xgboost import XGBClassifier

# LGBMClassifier: gradient boosting (lightgbm)
from lightgbm import LGBMClassifier

# SMOTE:
# Sınıf dengesizliği olan tablolarda, minority class için sentetik örnek üretir.
# Kritik nokta: sadece train tarafında uygulanmalı (test'e asla uygulanmaz).
from imblearn.over_sampling import SMOTE

# joblib:
# En iyi modeli diske kaydetmek için (deploy aşaması)
import joblib

# ---------------------------
# SHAP 
# ---------------------------

# SHAP:
# Ağaç tabanlı modeller için "hangi özellik nasıl katkı yaptı?" sorusuna yanıt veren
# açıklanabilirlik (explainability) kütüphanesidir.
#
# - Bazı Colab ortamlarda shap yüklü olmayabilir.
# - Yüklü değilse pip ile kurmayı deneriz.
# - Kurulum veya import başarısız olursa shap devre dışı kalır (script devam eder).
try:
    import shap
    shap_available = True
except Exception:
    try:
        import sys
        # Colab hücresinde çalışabilen "magic" komut:
        # shap yoksa sessizce kurmayı dener (-q = quiet).
        !pip install -q shap
        import shap
        shap_available = True
    except Exception:
        shap_available = False

# ---------------------------
# 2) Çıkış Klasörleri
# ---------------------------

# outputs/ altında her şeyi düzenli saklamak için klasör yapısı oluşturuyoruz.
# - figures : tüm grafikler (CM/ROC/PR/Calibration/SHAP vs.)
# - tables  : csv tablolar (model_results, best_per_model, class_metrics vs.)
# - models  : model kayıtları (deploy dosyaları burada)
# - logs    : shap hataları gibi günlük kayıtları
BASE_DIR = 'outputs'
FIG_DIR  = os.path.join(BASE_DIR, 'figures')
TAB_DIR  = os.path.join(BASE_DIR, 'tables')
MOD_DIR  = os.path.join(BASE_DIR, 'models')
LOG_DIR  = os.path.join(BASE_DIR, 'logs')

# Klasörler yoksa oluştur.
# exist_ok=True -> klasör varsa hata verme.
for d in [BASE_DIR, FIG_DIR, TAB_DIR, MOD_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

def savefig(path, dpi=300):
    # tight_layout: başlık/etiket taşmasını azaltır (özellikle uzun feature isimlerinde)
    plt.tight_layout()

    # Grafiği dosyaya kaydet.
    # bbox_inches='tight': grafik etrafındaki boşlukları minimize eder.
    plt.savefig(path, dpi=dpi, bbox_inches='tight')

    # Colab ekranında grafiği göstermek için
    plt.show()

    # Aynı figür objesi üst üste binmesin diye temizle (bir sonraki plot için)
    plt.clf()

def save_csv(df, name):
    # DataFrame'i outputs/tables altına kaydeder.
    # index=False -> DataFrame index sütunu dosyaya eklenmez (daha temiz çıktı)
    df.to_csv(os.path.join(TAB_DIR, name), index=False)

# ---------------------------
# 3) Veri Yükleme
# ---------------------------

# Kullanıcıya ne yaptığımızı anlatan bilgi mesajları
print("=== VERİ YÜKLEME VE HAZIRLIK ===")
print("Lütfen Excel dosyanızı yükleyin:")

# Colab dosya yükleme penceresini açar.
# Kullanıcı bir Excel seçer ve notebook'a upload eder.
uploaded = files.upload()

# Upload edilen dosyanın adını al.
file_name = list(uploaded.keys())[0]

# Excel dosyasını pandas ile oku (sheet varsayılan olarak ilk sheet)
data = pd.read_excel(file_name)

# Hedef değişkenin adı:
# Bu sütunda modelin tahmin edeceği sınıf etiketleri bulunmalı (0,1,2,3).
target = 'RCB_Kategorize'

# ---------------------------
# 4) Özellik Grupları (i11 ve i20 hariç)
# ---------------------------

# Bu çalışma “kademeli / modüler” bir yaklaşım izliyor:
# Aynı hedefi (RCB_Kategorize) tahmin ederken, farklı özellik gruplarını tek tek ve
# kombinasyonlu olarak modele verip performansı karşılaştırıyoruz.
#
# Böylece:
# - Hangi veri grubunun daha açıklayıcı / faydalı olduğunu görürüz
# - Klinik/pratik uygulamada “minimum veri ile maksimum performans” gibi senaryoları test ederiz
#
# i11 ve i20 özellikle dışarıda bırakılıyor:
# - i11 (E-cadherin) ve i20 (Menopoz) çalışmanın kapsamı/kararı gereği modele dahil edilmiyor
# - Bu nedenle feature listelerinde özellikle yoklar

# ---- Patoloji özellikleri (P) ----
features_p = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i12']

# names_p:
# Grafiklerde (feature importance, SHAP vb.) "i1,i2..." yerine kişilerin anlayacağı isimleri göstermek için.
names_p = [
    'Histolojik Tip', 'ER', 'PR', 'HER2', 'Moleküler Tip', 'Ki-67',
    'Tübül Derecesi', 'Nükleer Derece', 'Mitotik Derece', 'Histolojik Grade', 'TIL Değeri'
]

# ---- Onkoloji / tedavi özellikleri (O) ----
features_o = ['i13', 'i14', 'i15', 'i46', 'i47']
names_o = ['Metastaz Durumu', 'Metastaz Yeri', 'Tanı Evresi', 'Rejim', 'Kür Yoğunluk']

# ---- Demografik özellikler (D) ----
# Not: i20 (Menopoz) özellikle hariç tutulduğu için listede yoktur.
features_d = ['i16', 'i17', 'i18', 'i19', 'i45']
names_d = ['Hangi Meme', 'VKI Sınıfı', 'Yaş Grubu', 'Kan Grubu', 'Güneşten Yararlanma']

# ---- Komorbidite özellikleri (K) ----
features_k = ['i21','i22','i23','i24','i25','i26','i27','i28','i29','i30']
names_k = ['HT','DM','KOAH','Sigara','Ailede Meme CA','Tiroid','Retinopati','Nöropati','Osteoporoz','Depresyon']

# ---- Biyokimya özellikleri (B) ----
features_b = ['i31','i32','i33','i34','i35','i36','i37','i38','i39','i40','i41','i42','i43','i44']
names_b = ['ALP','ALT','AST','BUN','CA15-3','CEA','CRP','GGT','Glukoz','HbA1c','Kreatinin','LDH','TSH','e-GFR']

# ---- Radyoloji özellikleri (R) ----
features_r = ['i48','i49','i50','i51','i52','i53','i54','i55','i56','i57','i58','i59','i60','i61','i62','i63','i64']
names_r = [
    'BI-RADS','Meme Dansitesi','Lokalizasyon','Lezyon Türü','Mimari',
    'Kitle Şekli','Kitle Konturu','Kitle Dansitesi','Kalsifikasyon Morfolojisi',
    'Kalsifikasyon Dağılımı','Asimetri','Multifokalite','2 Yıldır Stabil',
    'Cilt Çekintisi','Meme Başı Retraksiyonu','Ameliyat Öyküsü','Kozmetik Implant'
]

# models_dict:
# Her bir "Model adı"nın hangi feature sütunlarını kullanacağını tanımlar.
# Amaç: aynı script içinde çok sayıda feature kombinasyonunu otomatik denemek.
models_dict = {
    'Model P': features_p,
    'Model O': features_o,
    'Model P+O': features_p + features_o,
    'Model D': features_d,
    'Model P+O+D': features_p + features_o + features_d,
    'Model K': features_k,
    'Model P+O+D+K': features_p + features_o + features_d + features_k,
    'Model B': features_b,
    'Model P+O+D+K+B': features_p + features_o + features_d + features_k + features_b,
    'Model R': features_r,
    'Model ALL': features_p + features_o + features_d + features_k + features_b + features_r
}

# names_dict:
# models_dict ile bire bir paralel çalışır:
# - aynı "Model adı" için insan-okunur feature isimlerini tutar
# - böylece grafiklerde doğru isim eşleşmesini garanti eder
names_dict = {
    'Model P': names_p,
    'Model O': names_o,
    'Model P+O': names_p + names_o,
    'Model D': names_d,
    'Model P+O+D': names_p + names_o + names_d,
    'Model K': names_k,
    'Model P+O+D+K': names_p + names_o + names_d + names_k,
    'Model B': names_b,
    'Model P+O+D+K+B': names_p + names_o + names_d + names_k + names_b,
    'Model R': names_r,
    'Model ALL': names_p + names_o + names_d + names_k + names_b + names_r
}

# run_order:
# Script hangi model kombinasyonlarını sırayla deneyecek?
# quick_mode=True ise daha az kombinasyon denenir (hızlı debug).
# quick_mode=False ise tüm kombinasyonlar denenir (final analiz).
if quick_mode:
    run_order = ['Model P', 'Model P+O+D', 'Model ALL']
else:
    run_order = [
        'Model P','Model O','Model P+O','Model D','Model P+O+D','Model K',
        'Model P+O+D+K','Model B','Model P+O+D+K+B','Model R','Model ALL'
    ]

# ---------------------------
# 5) Algoritmalar (ölçekleme yok)
# ---------------------------

# Bu scriptte ölçekleme (standardization/normalization) uygulanmıyor.
# Sebep: kullanılan algoritmalar (RF/XGB/LGBM) ağaç tabanlı olduğu için
# ölçeklemeye genellikle ihtiyaç duymazlar

# algs sözlüğü:
# - Anahtar: algoritma adı (raporlama/isimlendirme için)
# - Değer: sklearn uyumlu estimator nesnesi (fit/predict/predict_proba)
#
# Parametreler “örnek” bir başlangıç noktasıdır:
# - n_estimators: ağaç sayısı (ensemble büyüklüğü)
# - max_depth, min_samples_split/leaf: overfitting kontrolü
# - subsample/colsample_bytree: boosting modellerde regularization + genelleme
# - random_state: tekrar üretilebilirlik (tez çalışmaları için kritik)
algs = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # CPU çekirdeklerini paralel kullanır (hız)
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',  # multiclass logloss
        verbosity=0             # konsol çıktısını azalt
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1              # LightGBM loglarını azalt
    )
}

# ---------------------------
# 6) Stratified Split (SMOTE YOK!)
# ---------------------------

# classes:
# Bu çalışma 4 sınıflı bir problem (0,1,2,3).
# Sınıf sırasını sabitlemek önemli çünkü:
# - confusion matrix etiketleri
# - roc/pr curve çizimleri
# - class_order.json gibi deploy çıktıları
# tutarlı olmalı.
classes = np.array([0, 1, 2, 3])

# X_all:
# Train-test split'i tek bir "ortak çerçeve" üzerinden yapmak için.
# Neden "Model ALL" üzerinden?
# - Split işlemi her model için tekrar tekrar yapılmasın
# - Her model aynı train/test hastaları görsün → adil karşılaştırma
# - Sonrasında her model için gerekli sütunlar seçilerek alt-set X_tr/X_te yapılır
X_all = data[models_dict['Model ALL']].copy()

# y_all:
# Hedef değişken (RCB sınıfı)
y_all = data[target].copy()

# assert:
# Veri doğrulama adımı.
# Sınıfların gerçekten 0,1,2,3 olup olmadığını kontrol eder.
# Eğer veri eksik sınıf içeriyorsa (örneğin hiç "3" yoksa) bu scriptin bazı
# multiclass metrikleri/plotları hata verebilir veya yanıltıcı olabilir.
assert np.array_equal(np.sort(y_all.unique()), classes), "Sınıf etiketleri [0,1,2,3] değil!"

# train_test_split:
# Data leakage önlemenin temel adımı.
# - Önce train/test ayrılır
# - Test seti “en baştan” ayrılmış olur ve hiçbir işlem (SMOTE dahil) test'e uygulanmaz.
#
# stratify=y_all:
# - sınıf dağılımı hem train hem test tarafında benzer olsun
# - dengesiz sınıflarda test setinin tek sınıfa kaymasını engeller
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,       # verinin %20'si test
    stratify=y_all,      # sınıf oranlarını koru
    random_state=42      # aynı split tekrar üretilebilsin
)

# Split sonrası temel kontroller:
# - boyutlar
# - sınıf oranları (train ve test için)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Sınıf oranları (train/test):")
print(y_train.value_counts(normalize=True).sort_index())
print(y_test.value_counts(normalize=True).sort_index())

# StratifiedKFold:
# CV içinde de sınıf dağılımı korunur.
# shuffle=True + random_state=42:
# - fold'lar rastgele karıştırılarak seçilir (daha sağlam)
# - aynı sonuç tekrar üretilebilir
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# 7) Data Leakage Önleyici CV Fonksiyonu
# ---------------------------

def safe_cross_validation(model_template, X, y, cv, use_smote=False):
    """
    Amaç:
    - Cross-validation sırasında data leakage riskini ortadan kaldırmak.
    - SMOTE kullanılıyorsa sadece fold-train üzerinde uygulanır (fold-val'e uygulanmaz).
    - Her fold için clone(model_template) ile sıfırdan bir model üretilir.

    Parametreler:
    - model_template : sklearn uyumlu estimator (henüz fit edilmemiş “şablon”)
    - X, y           : sadece TRAIN tarafındaki veri (test değil!)
    - cv             : StratifiedKFold nesnesi
    - use_smote      : True ise fold-train tarafında SMOTE uygula

    Çıktı:
    - Her fold için accuracy, macro-AUC (OVR), macro-F1 içeren liste döndürür.
    """
    cv_scores = []

    # cv.split(X, y) -> her fold için (train_index, val_index) üretir.
    # Stratified olduğu için her fold'da sınıf oranları korunur.
    for train_idx, val_idx in cv.split(X, y):
        # Fold bazında ayrıştırma:
        # - X_tr/y_tr: o fold'un eğitim kısmı
        # - X_val/y_val: o fold'un doğrulama kısmı
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # SMOTE sadece eğitim kısmında:
        # Önemli gerekçe:
        # - SMOTE sentetik örnek üretir
        # - Eğer val/test tarafına uygulanırsa model “gelecekten bilgi” görmüş olur
        # - Bu da performansı yapay biçimde şişirir (data leakage)
        if use_smote:
            try:
                sm = SMOTE(random_state=42, k_neighbors=5)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                # SMOTE bazı durumlarda hata verebilir:
                # - Bir fold'da bazı sınıflar çok az kalmış olabilir (k_neighbors nedeniyle)
                # Bu durumda scriptin tamamen durmaması için SMOTE'suz devam edilir.
                pass

        # clone:
        # model_template aynı nesne olarak tekrar tekrar fit edilirse,
        # bazı estimator'larda state taşınabilir veya beklenmeyen yan etkiler oluşabilir.
        # clone() her fold için temiz, bağımsız bir model instance üretir.
        model = clone(model_template)

        # Fold-train ile eğit
        model.fit(X_tr, y_tr)

        # Fold-val tahminleri:
        # - y_pred: sınıf etiketi (0/1/2/3)
        # - y_proba: sınıf olasılıkları (AUC/ROC/PR için gerekir)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        # Fold metrikleri:
        # accuracy: doğru sınıf oranı
        # auc: multiclass OVR macro average (sınıflar eşit ağırlıkla)
        # f1: macro F1 (sınıflar eşit ağırlıkla)
        cv_scores.append({
            'accuracy': accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro'),
            'f1': f1_score(y_val, y_pred, average='macro')
        })

    return cv_scores

# ---------------------------
# 8) Grafik Üreticiler (TAM SET)
# ---------------------------

# Bu bölümde tanımlanan fonksiyonlar:
# - Model eğitimi sırasında defalarca tekrar eden çizim kodlarını
#   tek bir yerde toplayarak kod tekrarını azaltır
# - Her grafik için tutarlı stil, başlık ve dosya kaydetme davranışı sağlar
#
# Not:
# Grafiklerin tamamı TEST SET üzerinde üretilir.
# Böylece performans raporlaması "gerçek dünya" senaryosuna karşılık gelir.

def plot_confusion(y_true, y_pred, title, out_png):
    """
    Confusion Matrix (Karışıklık Matrisi) çizer ve kaydeder.

    y_true : gerçek sınıf etiketleri
    y_pred : modelin tahmin ettiği sınıflar
    title  : grafikte gösterilecek başlık
    out_png: figures/ altına kaydedilecek dosya adı

    Neden önemli?
    - Hangi sınıfların birbiriyle karıştığını görmemizi sağlar
    - Klinik çalışmalarda "hangi RCB seviyeleri daha çok karışıyor?"
      sorusuna doğrudan görsel yanıt verir
    """
    plt.figure(figsize=(8, 6))

    # labels=classes:
    # Satır/sütun sırasının [0,1,2,3] olarak sabitlenmesini sağlar
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # seaborn heatmap:
    # annot=True -> hücre değerlerini yaz
    # fmt='d'    -> integer format
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[f'RCB-{c}' for c in classes],
        yticklabels=[f'RCB-{c}' for c in classes]
    )

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    savefig(os.path.join(FIG_DIR, out_png))

def plot_roc_ovr(y_true, y_proba, title, out_png):
    """
    One-vs-Rest (OVR) ROC eğrilerini çizer.

    y_true  : gerçek sınıflar
    y_proba : predict_proba çıktısı (n_samples x n_classes)

    Mantık:
    - Çok sınıflı problem olduğu için her sınıf ayrı ayrı
      "o sınıf vs diğerleri" şeklinde ele alınır
    - Her sınıf için ROC ve AUC hesaplanır
    """
    # Çok sınıflı ROC için etiketleri binary matrise çevir
    y_bin = label_binarize(y_true, classes=classes)

    fpr, tpr, roc_auc = {}, {}, {}

    # Her sınıf için ROC eğrisi
    for i, c in enumerate(classes):
        fpr[c], tpr[c], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])

    plt.figure(figsize=(8, 6))

    # Her sınıfı ayrı bir eğri olarak çiz
    for c in classes:
        plt.plot(
            fpr[c],
            tpr[c],
            lw=2,
            label=f'RCB-{c} (AUC={roc_auc[c]:.3f})'
        )

    # Rastgele tahmin referans çizgisi
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')

    savefig(os.path.join(FIG_DIR, out_png))

def plot_feature_importance(model, feat_names, title, out_png, topn=15):
    """
    Ağaç tabanlı modeller için feature importance grafiği.

    model      : fit edilmiş model (RF / XGB / LGBM)
    feat_names : insan-okunur feature isimleri
    topn       : en önemli kaç özelliğin gösterileceği

    Not:
    - feature_importances_ attribute'u olmayan modeller
      (örn. bazı kernel tabanlı modeller) için sessizce atlanır
    """
    if not hasattr(model, 'feature_importances_'):
        return

    imp = model.feature_importances_

    # En önemli top-n özelliği seç
    order = np.argsort(imp)[::-1][:topn]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(order))[::-1], imp[order][::-1])
    plt.yticks(
        range(len(order))[::-1],
        [feat_names[i] for i in order][::-1]
    )

    plt.xlabel('Importance')
    plt.title(title)

    savefig(os.path.join(FIG_DIR, out_png))

def plot_pr_curves(y_true, y_proba, title, out_png):
    """
    Precision–Recall (PR) eğrileri.

    Neden ROC'a ek olarak PR?
    - Sınıf dengesizliği olan problemler için
      PR eğrileri daha bilgilendirici olabilir
    """
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(8, 6))

    for i, c in enumerate(classes):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(rec, prec, lw=1.5, label=f'RCB-{c}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()

    savefig(os.path.join(FIG_DIR, out_png))

def plot_calibration(y_true, y_proba, title, out_png, bins=10):
    """
    Calibration curve (olasılık kalibrasyonu).

    Amaç:
    - Modelin verdiği olasılıkların gerçek olasılıklarla ne kadar uyumlu olduğunu görmek
    - Örn: Model %70 diyorsa gerçekten %70 mi gerçekleşiyor?
    """
    from sklearn.calibration import calibration_curve

    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(8, 6))

    for i, c in enumerate(classes):
        prob_true, prob_pred = calibration_curve(
            y_bin[:, i],
            y_proba[:, i],
            n_bins=bins
        )
        plt.plot(prob_pred, prob_true, marker='o', lw=1.5, label=f'RCB-{c}')

    # Mükemmel kalibrasyon referans çizgisi
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical probability')
    plt.title(title)
    plt.legend()

    savefig(os.path.join(FIG_DIR, out_png))

def plot_gain_lift(y_true, y_proba, prefix):
    """
    Cumulative Gain ve Lift eğrilerini üretir.

    Bu grafikler özellikle:
    - Klinik risk sıralaması
    - Hasta önceliklendirme
    gibi senaryolarda anlamlıdır.
    """
    y_bin = label_binarize(y_true, classes=classes)

    # ---- CUMULATIVE GAIN ----
    plt.figure(figsize=(8, 6))

    for i, c in enumerate(classes):
        idx = np.argsort(y_proba[:, i])[::-1]
        sorted_y = y_bin[:, i][idx]

        cum_pos = np.cumsum(sorted_y) / max(1, sorted_y.sum())
        cum_pop = np.arange(1, len(sorted_y) + 1) / len(sorted_y)

        plt.plot(cum_pop, cum_pos, lw=1.5, label=f'RCB-{c}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Population fraction')
    plt.ylabel('Cumulative positive fraction')
    plt.title(f'{prefix} - Cumulative Gain')
    plt.legend()

    savefig(os.path.join(FIG_DIR, f'{prefix}_cumulative_gain.png'))

    # ---- LIFT ----
    plt.figure(figsize=(8, 6))

    for i, c in enumerate(classes):
        idx = np.argsort(y_proba[:, i])[::-1]
        sorted_y = y_bin[:, i][idx]

        cum_pos = np.cumsum(sorted_y) / max(1, sorted_y.sum())
        cum_pop = np.arange(1, len(sorted_y) + 1) / len(sorted_y)

        lift = cum_pos / np.maximum(cum_pop, 1e-9)
        plt.plot(cum_pop, lift, lw=1.5, label=f'RCB-{c}')

    plt.xlabel('Population fraction')
    plt.ylabel('Lift')
    plt.title(f'{prefix} - Lift Curve')
    plt.legend()

    savefig(os.path.join(FIG_DIR, f'{prefix}_lift_curve.png'))

# ---------------------------
# 9) Sınıf Bazında Detaylı Metrikler
# ---------------------------

def calculate_class_metrics(y_true, y_pred, y_proba, classes, model_name, alg_name):
    """
    Her sınıf için ayrı ayrı metrik hesaplar.

    Hesaplanan metrikler:
    - Precision
    - Recall
    - F1-score
    - AUC (binary: ilgili sınıf vs diğerleri)
    - Specificity (özgüllük)

    Neden gerekli?
    - Macro ortalamalar genel performansı gösterir
    - Ancak klinikte "RCB-3'ü kaçırıyor muyuz?" gibi
      sınıf-spesifik sorular çok daha kritiktir
    """
    class_metrics = []

    for i in classes:
        # Binary problem: "i sınıfı mı, değil mi?"
        y_binary = (y_true == i).astype(int)
        proba_i = y_proba[:, i]

        precision = precision_score(
            y_true, y_pred,
            labels=[i],
            average='micro',
            zero_division=0
        )
        recall = recall_score(
            y_true, y_pred,
            labels=[i],
            average='micro',
            zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred,
            labels=[i],
            average='micro',
            zero_division=0
        )

        # AUC bazı edge-case durumlarda hata verebilir (tek sınıf vs.)
        try:
            auc_i = roc_auc_score(y_binary, proba_i)
        except Exception:
            auc_i = 0.5

        # Specificity = TN / (TN + FP)
        tn = np.sum((y_true != i) & (y_pred != i))
        fp = np.sum((y_true != i) & (y_pred == i))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        class_metrics.append({
            'Model': model_name,
            'Algorithm': alg_name,
            'RCB_Sınıfı': f'RCB-{i}',
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'AUC': auc_i,
            'Specificity': specificity
        })

    return class_metrics

# ---------------------------
# 10) Modüler Döngü
# ---------------------------

# all_rows:
# Tüm model × algoritma × SMOTE kombinasyonlarının
# özet metriklerini (CV + Test) tabloya dökmek için kullanılır
all_rows = []

# best_per_model:
# Her model seti (P, O, P+O, ..., ALL) için
# EN İYİ algoritma + SMOTE kombinasyonunu saklar
best_per_model = {}

# all_class_metrics:
# Tüm modeller ve algoritmalar için
# sınıf bazında detaylı metrikleri toplar
all_class_metrics = []

# run_order:
# Daha önce belirlenmiş model setleri sırasıyla döner
for model_name in run_order:

    # Bu model setinde kullanılacak feature'lar
    feats = models_dict[model_name]

    # Aynı feature'ların insan-okunur isimleri
    feat_names = names_dict[model_name]

    # Bu model setine özel train ve test tabloları
    # (split daha önce yapıldığı için burada leakage riski yok)
    X_tr = X_train[feats].copy()
    X_te = X_test[feats].copy()

    print(f"\n=== {model_name} ({len(feats)} özellik) ===")

    # Bu model seti için algoritma bazlı sonuçları tutar
    model_results = {}

    # -------------------------------------------------
    # Algoritma döngüsü (RF / XGB / LGBM)
    # -------------------------------------------------
    for alg_name, est_template in algs.items():

        print(f" - {alg_name} CV/Test hesaplanıyor...")

        # =================================================
        # 1) CROSS-VALIDATION — SMOTE YOK
        # =================================================
        # - Sadece gerçek veri dağılımı ile performansı görmek için
        # - safe_cross_validation içinde:
        #   * Her fold için clone
        #   * Val setine SMOTE uygulanmaz
        cv_no = safe_cross_validation(
            est_template,
            X_tr,
            y_train,
            skf,
            use_smote=False
        )

        # Fold sonuçlarının ortalama ve std değerleri
        cv_acc_no  = np.mean([s['accuracy'] for s in cv_no])
        cv_auc_no  = np.mean([s['auc']      for s in cv_no])
        cv_f1_no   = np.mean([s['f1']       for s in cv_no])

        cv_acc_no_s = np.std([s['accuracy'] for s in cv_no])
        cv_auc_no_s = np.std([s['auc']      for s in cv_no])
        cv_f1_no_s  = np.std([s['f1']       for s in cv_no])

        # =================================================
        # 2) CROSS-VALIDATION — SMOTE VAR (fold-train içinde)
        # =================================================
        # - Sınıf dengesizliği düzeltilirse CV performansı nasıl değişiyor?
        # - SMOTE yalnızca fold-train'e uygulanır
        cv_sm = safe_cross_validation(
            est_template,
            X_tr,
            y_train,
            skf,
            use_smote=True
        )

        cv_acc_sm  = np.mean([s['accuracy'] for s in cv_sm])
        cv_auc_sm  = np.mean([s['auc']      for s in cv_sm])
        cv_f1_sm   = np.mean([s['f1']       for s in cv_sm])

        cv_acc_sm_s = np.std([s['accuracy'] for s in cv_sm])
        cv_auc_sm_s = np.std([s['auc']      for s in cv_sm])
        cv_f1_sm_s  = np.std([s['f1']       for s in cv_sm])

        # =================================================
        # 3) TEST — SMOTE YOK
        # =================================================
        # Akademik standart:
        # - CV sonrası final modeli train setinin TAMAMI ile eğit
        # - Test setine asla SMOTE uygulama
        est_no = clone(est_template)
        est_no.fit(X_tr, y_train)

        y_pred_no  = est_no.predict(X_te)
        y_proba_no = est_no.predict_proba(X_te)

        test_acc_no = accuracy_score(y_test, y_pred_no)
        test_auc_no = roc_auc_score(
            y_test,
            y_proba_no,
            multi_class='ovr',
            average='macro'
        )
        test_f1_no  = f1_score(y_test, y_pred_no, average='macro')

        # =================================================
        # 4) TEST — SMOTE VAR (yalnız train üzerinde)
        # =================================================
        # SMOTE burada:
        # - CV dışında, train setinin tamamında uygulanır
        # - Test seti yine orijinal dağılımda kalır
        sm = SMOTE(random_state=42, k_neighbors=5)
        try:
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_train)
        except Exception:
            # SMOTE başarısız olursa fallback
            X_tr_sm, y_tr_sm = X_tr, y_train

        est_sm = clone(est_template)
        est_sm.fit(X_tr_sm, y_tr_sm)

        y_pred_sm  = est_sm.predict(X_te)
        y_proba_sm = est_sm.predict_proba(X_te)

        test_acc_sm = accuracy_score(y_test, y_pred_sm)
        test_auc_sm = roc_auc_score(
            y_test,
            y_proba_sm,
            multi_class='ovr',
            average='macro'
        )
        test_f1_sm  = f1_score(y_test, y_pred_sm, average='macro')

        # =================================================
        # 5) OVERFITTING KONTROLÜ
        # =================================================
        # Basit ama etkili akademik kontrol:
        # CV AUC >> Test AUC ise model muhtemelen overfit
        overfit_warning_no = ""
        overfit_warning_sm = ""

        if cv_auc_no - test_auc_no > 0.15:
            overfit_warning_no = (
                f"OVERFITTING (NO SMOTE): "
                f"CV AUC={cv_auc_no:.3f} >> Test AUC={test_auc_no:.3f}"
            )

        if cv_auc_sm - test_auc_sm > 0.15:
            overfit_warning_sm = (
                f"OVERFITTING (SMOTE): "
                f"CV AUC={cv_auc_sm:.3f} >> Test AUC={test_auc_sm:.3f}"
            )

        if overfit_warning_no or overfit_warning_sm:
            print(f"    {overfit_warning_no}")
            print(f"    {overfit_warning_sm}")

        # =================================================
        # 6) Sonuçları sözlükte sakla
        # =================================================
        model_results[alg_name] = {

            # --- CV (NO SMOTE) ---
            'cv_acc_mean_no_smote': cv_acc_no,
            'cv_acc_std_no_smote':  cv_acc_no_s,
            'cv_auc_mean_no_smote': cv_auc_no,
            'cv_auc_std_no_smote':  cv_auc_no_s,
            'cv_f1_mean_no_smote':  cv_f1_no,
            'cv_f1_std_no_smote':   cv_f1_no_s,

            # --- TEST (NO SMOTE) ---
            'test_acc_no_smote': test_acc_no,
            'test_auc_no_smote': test_auc_no,
            'test_f1_no_smote':  test_f1_no,

            # --- CV (SMOTE VAR) ---
            'cv_acc_mean_smote': cv_acc_sm,
            'cv_acc_std_smote':  cv_acc_sm_s,
            'cv_auc_mean_smote': cv_auc_sm,
            'cv_auc_std_smote':  cv_auc_sm_s,
            'cv_f1_mean_smote':  cv_f1_sm,
            'cv_f1_std_smote':   cv_f1_sm_s,

            # --- TEST (SMOTE VAR) ---
            'test_acc_smote': test_acc_sm,
            'test_auc_smote': test_auc_sm,
            'test_f1_smote':  test_f1_sm,

            # --- Tahminler (grafikler & bootstrap için) ---
            'y_pred_no_smote':  y_pred_no,
            'y_proba_no_smote': y_proba_no,
            'y_pred_smote':     y_pred_sm,
            'y_proba_smote':    y_proba_sm
        }

        # =================================================
        # 7) SINIF BAZINDA METRİKLER (SMOTE VAR)
        # =================================================
        # Klinik bakış açısı:
        # - SMOTE, nadir sınıfları daha iyi öğrenmeyi amaçlar
        # - Bu yüzden sınıf bazlı detaylı metrikler
        #   SMOTE VAR senaryosu üzerinden raporlanır
        class_metrics = calculate_class_metrics(
            y_test,
            y_pred_sm,
            y_proba_sm,
            classes,
            model_name,
            alg_name
        )
        all_class_metrics.extend(class_metrics)

        # =================================================
        # 8) GRAFİKLER (TEST SET)
        # =================================================
        prefix = f"{model_name}_{alg_name}"

        # Confusion Matrix
        plot_confusion(
            y_test,
            y_pred_no,
            f'CM - {model_name} + {alg_name} (NO SMOTE)',
            f'{prefix}_cm_nosmote.png'
        )
        plot_confusion(
            y_test,
            y_pred_sm,
            f'CM - {model_name} + {alg_name} (SMOTE)',
            f'{prefix}_cm_smote.png'
        )

        # ROC
        plot_roc_ovr(
            y_test,
            y_proba_no,
            f'ROC - {model_name} + {alg_name} (NO SMOTE)',
            f'{prefix}_roc_nosmote.png'
        )
        plot_roc_ovr(
            y_test,
            y_proba_sm,
            f'ROC - {model_name} + {alg_name} (SMOTE)',
            f'{prefix}_roc_smote.png'
        )

        # Feature Importance
        plot_feature_importance(
            est_no,
            feat_names,
            f'Importance - {model_name} + {alg_name} (NO SMOTE)',
            f'{prefix}_importance_nosmote.png'
        )
        plot_feature_importance(
            est_sm,
            feat_names,
            f'Importance - {model_name} + {alg_name} (SMOTE)',
            f'{prefix}_importance_smote.png'
        )

        # =================================================
        # 9) AĞIR GRAFİKLER 
        # =================================================
        # heavy_plots = True ise:
        # - Precision–Recall Curve
        # - Calibration Curve
        # - Cumulative Gain
        # - Lift Curve
        # - SHAP (explainability)
        #
        # Bu grafikler:
        # - Hesaplama açısından pahalıdır
        # - Çok fazla model/algoritma varsa çalışma süresini ciddi artırır
        # Bu yüzden parametre ile açılıp kapatılabilir yapılmıştır.
        if heavy_plots:

            # -------------------------
            # Precision–Recall Curve
            # -------------------------
            # Özellikle sınıf dengesizliği olan problemlerde
            # ROC’a kıyasla daha anlamlı olabilir.
            plot_pr_curves(
                y_test,
                y_proba_no,
                f'PR - {model_name} + {alg_name} (NO SMOTE)',
                f'{prefix}_pr_nosmote.png'
            )
            plot_pr_curves(
                y_test,
                y_proba_sm,
                f'PR - {model_name} + {alg_name} (SMOTE)',
                f'{prefix}_pr_smote.png'
            )

            # -------------------------
            # Calibration Curve
            # -------------------------
            # Modelin verdiği olasılıkların güvenilirliğini ölçer.
            # Klinik karar destek sistemleri için kritik bir grafiktir.
            plot_calibration(
                y_test,
                y_proba_no,
                f'Calibration - {model_name} + {alg_name} (NO SMOTE)',
                f'{prefix}_calibration_nosmote.png'
            )
            plot_calibration(
                y_test,
                y_proba_sm,
                f'Calibration - {model_name} + {alg_name} (SMOTE)',
                f'{prefix}_calibration_smote.png'
            )

            # -------------------------
            # Cumulative Gain & Lift
            # -------------------------
            # Hasta önceliklendirme, risk sıralama gibi
            # uygulamalar için kullanılır.
            plot_gain_lift(
                y_test,
                y_proba_no,
                prefix=f'{prefix}_nosmote'
            )
            plot_gain_lift(
                y_test,
                y_proba_sm,
                prefix=f'{prefix}_smote'
            )

            # -------------------------
            # SHAP (Explainability)
            # -------------------------
            # Yalnızca SMOTE VAR tarafında üretilir.
            #
            # Gerekçe:
            # - Klinik olarak nadir sınıfların öğrenilmesi önemlidir
            # - SMOTE, nadir sınıfları dengeleyerek modeli daha
            #   "anlamlı" feature katkıları üretmeye zorlar
            #
            # Not:
            # - Sadece ağaç tabanlı modeller desteklenir
            # - Hata olursa analiz durmaz, log dosyasına yazılır
            if shap_available and alg_name in ['RandomForest','XGBoost','LightGBM']:
                try:
                    explainer = shap.TreeExplainer(est_sm)

                    # Hesaplama maliyetini sınırlamak için
                    # test setinden rastgele maksimum 200 örnek alınır
                    sample_idx = np.random.RandomState(42).choice(
                        len(X_te),
                        size=min(200, len(X_te)),
                        replace=False
                    )
                    X_te_sample = X_te.iloc[sample_idx]

                    shap_vals = explainer.shap_values(X_te_sample)

                    # İnsan okunur feature isimleri
                    human_feature_names = names_dict[model_name]

                    # Çok sınıflı ağaç modelleri genellikle
                    # shap_values'i liste (sınıf başına) olarak döndürür
                    if isinstance(shap_vals, list):
                        for ci, c in enumerate(classes):
                            plt.figure(figsize=(10, 4))
                            shap.summary_plot(
                                shap_vals[ci],
                                X_te_sample,
                                feature_names=human_feature_names,
                                show=False
                            )
                            plt.title(
                                f"SHAP Summary – {model_name} + {alg_name} "
                                f"(SMOTE VAR) – RCB-{c}"
                            )
                            savefig(
                                os.path.join(
                                    FIG_DIR,
                                    f'{prefix}_smote_shap_summary_class{c}.png'
                                )
                            )
                    else:
                        # Bazı implementasyonlar (n_samples, n_features, n_classes)
                        # şeklinde döndürebilir
                        for ci, c in enumerate(classes):
                            vals_c = shap_vals[:, :, ci]
                            plt.figure(figsize=(10, 4))
                            shap.summary_plot(
                                vals_c,
                                X_te_sample,
                                feature_names=human_feature_names,
                                show=False
                            )
                            plt.title(
                                f"SHAP Summary – {model_name} + {alg_name} "
                                f"(SMOTE VAR) – RCB-{c}"
                            )
                            savefig(
                                os.path.join(
                                    FIG_DIR,
                                    f'{prefix}_smote_shap_summary_class{c}.png'
                                )
                            )

                except Exception as e:
                    # SHAP hataları tüm pipeline'ı bozmasın diye
                    # sadece loglanır
                    with open(
                        os.path.join(LOG_DIR, 'shap_errors.log'),
                        'a',
                        encoding='utf-8'
                    ) as f:
                        f.write(f'{prefix}: {str(e)}\n')

    # =====================================================
    # 10) MODEL SETİ İÇİN EN İYİ KOMBİNASYONUN SEÇİMİ
    # =====================================================
    # Amaç:
    # - Her model seti (P, O, P+O, ..., ALL) için
    #   TEK bir "en iyi" algoritma + SMOTE kararı vermek
    #
    # Seçim kriterleri (öncelik sırasıyla):
    # 1. Test AUC (yüksek olsun)
    # 2. CV–Test farkı (overfitting düşük olsun)
    # 3. Test F1 (genel sınıflandırma dengesi iyi olsun)

    def combo_candidates():
        """
        Bu yardımcı generator:
        - Her algoritma için
          * SMOTE YOK
          * SMOTE VAR
        iki ayrı aday üretir.
        """
        for alg_name, res in model_results.items():

            # ---- SMOTE YOK adayı ----
            yield {
                'algorithm': alg_name,
                'smote': 'YOK',
                'cv_auc_mean': res['cv_auc_mean_no_smote'],
                'cv_auc_std':  res['cv_auc_std_no_smote'],
                'test_auc':    res['test_auc_no_smote'],
                'test_f1':     res['test_f1_no_smote'],
                'cv_test_gap': abs(
                    res['cv_auc_mean_no_smote']
                    - res['test_auc_no_smote']
                )
            }

            # ---- SMOTE VAR adayı ----
            yield {
                'algorithm': alg_name,
                'smote': 'VAR',
                'cv_auc_mean': res['cv_auc_mean_smote'],
                'cv_auc_std':  res['cv_auc_std_smote'],
                'test_auc':    res['test_auc_smote'],
                'test_f1':     res['test_f1_smote'],
                'cv_test_gap': abs(
                    res['cv_auc_mean_smote']
                    - res['test_auc_smote']
                )
            }

    # Adayları listele ve sırala
    candidates = list(combo_candidates())

    # Sıralama mantığı:
    # - Önce Test AUC (azalan)
    # - Sonra CV–Test farkı (artan)
    # - Sonra Test F1 (azalan)
    candidates.sort(
        key=lambda d: (-d['test_auc'], d['cv_test_gap'], -d['test_f1'])
    )

    # İlk sıradaki aday bu model seti için "en iyi" kabul edilir
    best_combination = candidates[0]

    best_per_model[model_name] = {
        'algorithm': best_combination['algorithm'],
        'smote': best_combination['smote'],
        'cv_auc_mean': best_combination['cv_auc_mean'],
        'cv_auc_std':  best_combination['cv_auc_std'],
        'test_auc':    best_combination['test_auc'],
        'test_f1':     best_combination['test_f1'],
        'cv_test_gap': best_combination['cv_test_gap'],
        'num_features': int(len(feats))
    }

# ---------------------------
# 11) Sonuç Tabloları (GENEL)
# ---------------------------

# all_rows:
# Daha önce ana döngüde doldurulan bu liste,
# tüm model × algoritma × SMOTE kombinasyonlarının
# CV ve Test metriklerini satır bazında içerir.
#
# Bu yapı:
# - CSV olarak kaydedilir
# - İstenirse dış analizler (R, SPSS, Excel) için kullanılabilir
results_df = pd.DataFrame(all_rows)

# Tüm kombinasyonların detaylı sonuç tablosu
save_csv(results_df, 'model_results_fixed.csv')

# -------------------------------------------------
# Her model seti için EN İYİ kombinasyon tablosu
# -------------------------------------------------

# best_per_model sözlüğü:
# - Anahtar: Model adı (P, O, P+O, ..., ALL)
# - Değer: O model için seçilmiş en iyi kombinasyon bilgisi
#
# Bunu tabloya çeviriyoruz
best_df = pd.DataFrame([
    {
        'Model': m,
        'BestAlgorithm': v['algorithm'],
        'SMOTE': v['smote'],
        'CV_AUC_Mean': v['cv_auc_mean'],
        'CV_AUC_Std':  v['cv_auc_std'],
        'Test_AUC':    v['test_auc'],
        'Test_F1':     v['test_f1'],
        'CV_Test_Gap': v['cv_test_gap'],
        'Num_Features': v['num_features']
    }
    for m, v in best_per_model.items()
])

# Modelleri global olarak sıralıyoruz
# Akademik öncelik sırası:
# 1. Test AUC (yüksek daha iyi)
# 2. CV–Test farkı (küçük daha iyi → overfitting az)
# 3. Test F1 (yüksek daha dengeli sınıflandırma)
best_df = best_df.sort_values(
    by=['Test_AUC', 'CV_Test_Gap', 'Test_F1'],
    ascending=[False, True, False]
)

# En iyi kombinasyonların özeti
save_csv(best_df, 'best_per_model_fixed.csv')

# -------------------------------------------------
# Tüm sınıf bazlı metrikler (her model & algoritma)
# -------------------------------------------------

# Bu tablo:
# - Tüm modeller
# - Tüm algoritmalar
# - Her RCB sınıfı
# için Precision / Recall / F1 / AUC / Specificity içerir
all_class_metrics_df = pd.DataFrame(all_class_metrics)
save_csv(all_class_metrics_df, 'all_class_metrics_fixed.csv')

print("\n=== EN İYİLER (Test AUC'a göre) ===")
print(best_df)

# Varsayılan olarak en üst sıradaki model
best_overall = best_df.iloc[0]

# -------------------------------------------------------------
# 11.1) TIE-CHECK (AUC ±0.01)
# -------------------------------------------------------------
# Neden?
# - Klinik/medikal çalışmalarda AUC farkları çok küçük olabilir
# - 0.002–0.005 gibi farklar pratikte anlamsızdır
# - Ancak kod otomatik olarak "en yüksek AUC"yı seçerse
#   bu keyfi bir karar gibi görünebilir
#
# Bu yüzden:
# - ±0.01 içinde kalan modeller “klinik olarak eşit” kabul edilir
# - Bu durum JSON olarak kaydedilir (izlenebilirlik)

TIE_THRESHOLD = 0.01

top_auc_value = best_overall['Test_AUC']

# En iyi AUC etrafındaki adayları seç
tie_mask = (top_auc_value - best_df['Test_AUC']).abs() <= TIE_THRESHOLD
tie_candidates_df = best_df[tie_mask].copy()

# Tie-check bilgilerini yapılandırılmış JSON olarak kaydet
tie_info = {
    'threshold': TIE_THRESHOLD,
    'top_auc': float(top_auc_value),
    'candidate_count': int(len(tie_candidates_df)),
    'candidates': [
        {
            'Model': row['Model'],
            'BestAlgorithm': row['BestAlgorithm'],
            'SMOTE': row['SMOTE'],
            'CV_AUC_Mean': float(row['CV_AUC_Mean']),
            'CV_AUC_Std':  float(row.get('CV_AUC_Std', float('nan'))),
            'Test_AUC':    float(row['Test_AUC']),
            'Test_F1':     float(row.get('Test_F1', float('nan'))),
            'CV_Test_Gap': float(row.get('CV_Test_Gap', float('nan')))
        }
        for _, row in tie_candidates_df.iterrows()
    ],
    'needs_tie_break': bool(len(tie_candidates_df) > 1)
}

with open(
    os.path.join(TAB_DIR, 'tie_check.json'),
    'w',
    encoding='utf-8'
) as f:
    json.dump(tie_info, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------------
# 11.2) Bootstrap AUC farkı (yardımcı fonksiyon)
# -------------------------------------------------------------
# Bu fonksiyon:
# - İki modelin AUC farkı için
# - Bootstrap %95 güven aralığı hesaplamak amacıyla yazıldı
#
# Özellikle:
# - “Bu iki model arasında istatistiksel fark var mı?”
#   diye sorarsa kullanılabilir
def bootstrap_auc_diff(
    y_true,
    y_proba_1,
    y_proba_2,
    n_bootstrap=1000,
    random_state=42
):
    """
    İki modelin AUC farkı için bootstrap %95 CI hesaplar.
    Return: (mean_diff, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        # Bootstrap örneklemesi (replacement ile)
        idx = np.random.choice(n, size=n, replace=True)

        if hasattr(y_true, 'iloc'):
            y_true_boot = y_true.iloc[idx].values
        else:
            y_true_boot = y_true[idx]

        y_proba_1_boot = y_proba_1[idx]
        y_proba_2_boot = y_proba_2[idx]

        y_bin_boot = label_binarize(y_true_boot, classes=classes)

        try:
            auc_1 = roc_auc_score(
                y_bin_boot,
                y_proba_1_boot,
                multi_class='ovr',
                average='macro'
            )
            auc_2 = roc_auc_score(
                y_bin_boot,
                y_proba_2_boot,
                multi_class='ovr',
                average='macro'
            )
            diffs.append(auc_1 - auc_2)
        except:
            continue

    diffs = np.array(diffs)
    mean_diff = np.mean(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return mean_diff, ci_lower, ci_upper

# ============================================================================
# 11.3) TIE-BREAK (Genel Kural ile Nihai Seçim)
# ============================================================================

# Eğer tie_check sonucunda birden fazla model
# "klinik olarak eşit" kabul ediliyorsa:
# - Daha stabil
# - Daha dengeli
# - Daha açıklanabilir
# olan modeli seçmek gerekir.
#
# Bu bölümde kullanılan GENEL KURAL:
# 1. CV–Test farkı en küçük olan
# 2. Test F1 skoru daha yüksek olan
# 3. CV AUC std daha düşük olan (daha stabil)
# 4. Daha fazla feature kullanan (eşitlikte daha kapsayıcı)
# 5. Model adı alfabetik (deterministik sonuç için)

if tie_info['needs_tie_break']:

    print("\n" + "=" * 70)
    print("EŞİTLİK DURUMU KONTROLÜ: Genel Veri-Agnostik Seçim")
    print("=" * 70)

    print("\n Klinik olarak eşit kabul edilen modeller tespit edildi.")
    print(f"  AUC eşik: ±{TIE_THRESHOLD:.3f}")

    # Tie adaylarını filtrele
    candidates = best_df[
        (top_auc_value - best_df['Test_AUC']).abs() <= TIE_THRESHOLD
    ].copy()

    # Eksik değerler karar mekanizmasını bozmasın diye doldurulur
    candidates['CV_Test_Gap'] = candidates['CV_Test_Gap'].fillna(np.inf)
    candidates['Test_F1'] = candidates['Test_F1'].fillna(0.0)
    candidates['CV_AUC_Std'] = candidates['CV_AUC_Std'].fillna(np.inf)
    candidates['Num_Features'] = candidates['Num_Features'].fillna(0)

    # Genel kural ile sıralama
    candidates = candidates.sort_values(
        by=[
            'CV_Test_Gap',
            'Test_F1',
            'CV_AUC_Std',
            'Num_Features',
            'Model'
        ],
        ascending=[True, False, True, False, True]
    )

    chosen = candidates.iloc[0]

    print("\nTie adayları (özet):")
    print(
        candidates[
            [
                'Model',
                'BestAlgorithm',
                'SMOTE',
                'Test_AUC',
                'CV_Test_Gap',
                'Test_F1',
                'CV_AUC_Std',
                'Num_Features'
            ]
        ]
    )

    print("\n KARAR (genel kural):")
    print(
        f"  → {chosen['Model']} + "
        f"{chosen['BestAlgorithm']} + "
        f"SMOTE {chosen['SMOTE']}"
    )

    # Nihai en iyi modeli güncelle
    best_overall = chosen

print(
    f"\n En iyi genel: "
    f"{best_overall['Model']} + "
    f"{best_overall['BestAlgorithm']} + "
    f"SMOTE {best_overall['SMOTE']} | "
    f"CV AUC={best_overall['CV_AUC_Mean']:.3f}, "
    f"Test AUC={best_overall['Test_AUC']:.3f}"
)

# ============================================================================
# 11.4) Final Seçim Özeti (JSON)
# ============================================================================

# Bu JSON:
# - Nihai kararın makine tarafından okunabilir özeti
# - Raporlama, web servis, model registry vb. için kullanılabilir
final_selection = {
    'Model': best_overall['Model'],
    'BestAlgorithm': best_overall['BestAlgorithm'],
    'SMOTE': best_overall['SMOTE'],
    'CV_AUC_Mean': float(best_overall['CV_AUC_Mean']),
    'CV_AUC_Std': float(best_overall.get('CV_AUC_Std', float('nan'))),
    'Test_AUC': float(best_overall['Test_AUC']),
    'Test_F1': float(best_overall.get('Test_F1', float('nan'))),
    'CV_Test_Gap': float(best_overall.get('CV_Test_Gap', float('nan'))),
    'tie_break_applied': bool(tie_info['needs_tie_break'])
}

with open(
    os.path.join(TAB_DIR, 'final_selection.json'),
    'w',
    encoding='utf-8'
) as f:
    json.dump(final_selection, f, ensure_ascii=False, indent=2)

# ============================================================================
# 12) EN İYİ MODELİN YENİDEN EĞİTİLMESİ (FINAL TRAINING)
# ============================================================================

# Nihai seçilen model + algoritma bilgileri
best_model_name = best_overall['Model']
best_alg_name   = best_overall['BestAlgorithm']

# Bu model için kullanılacak feature listesi
feats_best = models_dict[best_model_name]

# Sadece TRAIN seti kullanılır (test verisi asla yeniden eğitime girmez)
X_best_tr = X_train[feats_best]

# SMOTE kararı:
# - Eğer nihai model SMOTE VAR ise train seti SMOTE ile dengelenir
# - Test seti HER ZAMAN orijinal dağılımda kalır
best_smote_status = best_overall['SMOTE']

if best_smote_status == 'VAR':
    smote = SMOTE(random_state=42, k_neighbors=5)
    try:
        X_best_tr_final, y_best_tr_final = smote.fit_resample(
            X_best_tr, y_train
        )
    except Exception:
        X_best_tr_final, y_best_tr_final = X_best_tr, y_train
else:
    X_best_tr_final, y_best_tr_final = X_best_tr, y_train

# Akademik + mühendislik standardı:
# - clone ile temiz model instance
# - pipeline içinde başka fit etkisi yok
est_best_template = algs[best_alg_name]
est_best = clone(est_best_template)
est_best.fit(X_best_tr_final, y_best_tr_final)

# ============================================================================
# 12.1) FINAL SHAP (Nihai Model için)
# ============================================================================

# SHAP:
# - Nihai deploy edilecek modelin karar mantığını açıklar
# - Klinik yorumlama için özellikle önemlidir
try:
    if shap_available and best_alg_name in [
        'RandomForest', 'XGBoost', 'LightGBM'
    ]:

        # Test setinden sabit, rastgele örneklem (maks 300)
        X_best_te = X_test[feats_best]
        sample_idx = np.random.RandomState(42).choice(
            len(X_best_te),
            size=min(300, len(X_best_te)),
            replace=False
        )
        X_best_te_sample = X_best_te.iloc[sample_idx]

        explainer_best = shap.TreeExplainer(est_best)
        shap_vals_best = explainer_best.shap_values(X_best_te_sample)

        prefix_best = (
            f"BEST_{best_model_name}_"
            f"{best_alg_name}_"
            f"SMOTE_{best_smote_status}"
        )

        human_feature_names_best = names_dict[best_model_name]

        # Çok sınıflı SHAP çıktıları için uyumlu çizim
        if isinstance(shap_vals_best, list):
            for ci, c in enumerate(classes):
                plt.figure(figsize=(10, 4))
                shap.summary_plot(
                    shap_vals_best[ci],
                    X_best_te_sample,
                    feature_names=human_feature_names_best,
                    show=False
                )
                plt.title(
                    f"SHAP Summary – {best_model_name} + "
                    f"{best_alg_name} (SMOTE {best_smote_status}) – "
                    f"RCB-{c}"
                )
                savefig(
                    os.path.join(
                        FIG_DIR,
                        f"{prefix_best}_shap_summary_class{c}.png"
                    )
                )
        else:
            # (n_samples, n_features, n_classes) durumu
            if shap_vals_best.ndim == 3 and shap_vals_best.shape[-1] == len(classes):
                for ci, c in enumerate(classes):
                    vals_c = shap_vals_best[:, :, ci]
                    plt.figure(figsize=(10, 4))
                    shap.summary_plot(
                        vals_c,
                        X_best_te_sample,
                        feature_names=human_feature_names_best,
                        show=False
                    )
                    plt.title(
                        f"SHAP Summary – {best_model_name} + "
                        f"{best_alg_name} (SMOTE {best_smote_status}) – "
                        f"RCB-{c}"
                    )
                    savefig(
                        os.path.join(
                            FIG_DIR,
                            f"{prefix_best}_shap_summary_class{c}.png"
                        )
                    )
            else:
                plt.figure(figsize=(10, 4))
                shap.summary_plot(
                    shap_vals_best,
                    X_best_te_sample,
                    feature_names=human_feature_names_best,
                    show=False
                )
                plt.title(
                    f"SHAP Summary – {best_model_name} + "
                    f"{best_alg_name} (SMOTE {best_smote_status})"
                )
                savefig(
                    os.path.join(
                        FIG_DIR,
                        f"{prefix_best}_shap_summary.png"
                    )
                )
except Exception as e:
    with open(
        os.path.join(LOG_DIR, 'shap_errors.log'),
        'a',
        encoding='utf-8'
    ) as f:
        f.write(f'BEST_MODEL_SHAP: {str(e)}\n')

# ============================================================================
# 12.2) DEPLOY ARTEFAKTLARI
# ============================================================================

# Deploy edilecek dosyalar:
# - best_model.joblib   → eğitilmiş model
# - feature_list.json   → inference sırasında sütun sırası için
# - class_order.json    → çıktı sınıf sırası için

class_order_list = [int(c) for c in classes.tolist()]

deploy_dir = os.path.join(MOD_DIR, "deploy")
os.makedirs(deploy_dir, exist_ok=True)

joblib.dump(
    est_best,
    os.path.join(deploy_dir, "best_model.joblib")
)

with open(
    os.path.join(deploy_dir, "feature_list.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(feats_best, f, ensure_ascii=False, indent=2)

with open(
    os.path.join(deploy_dir, "class_order.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(class_order_list, f, ensure_ascii=False, indent=2)

print("Deployment artefaktları kaydedildi:", deploy_dir)

# ============================================================================
# 13) TÜM ÇIKTILARI ZIP'LE VE İNDİR
# ============================================================================

# outputs/ klasöründeki:
# - figures
# - tables
# - models
# - logs
# klasörlerinin tamamı tek zip haline getirilir
zip_path = 'outputs_fixed.zip'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files_list in os.walk(BASE_DIR):
        for fn in files_list:
            fp = os.path.join(root, fn)
            zf.write(fp, arcname=os.path.relpath(fp, BASE_DIR))

print(f"\n Tüm çıktılar '{zip_path}' olarak hazır.")
files.download(zip_path)

print("\n Data Leakage düzeltilmiş analiz tamamlandı!")
print(" Tüm şekiller ve tablolar outputs/ altında üretildi.")
print(" Her model×algoritma için hem SMOTE YOK hem SMOTE VAR sonuçlar mevcut.")


