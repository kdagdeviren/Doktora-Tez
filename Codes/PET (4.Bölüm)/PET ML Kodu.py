# =============================================================================
# PET ANALİZİ (RF + LightGBM + XGBoost) + ŞEKİLLER + EXCEL (LEAKAGE-SAFE SMOTE)
# =============================================================================
# Bu scriptin genel amacı:
# - PET verisi bulunan hastalar üzerinde RCB_Kategorize (çok sınıflı) tahmin modeli kurmak.
# - Üç farklı feature setini karşılaştırmak:
#   1) Sadece PET (Model PET)
#   2) Sadece klinik/patoloji/radyoloji vb. (Model ALL)
#   3) ALL + PET birleşik (Model ALL + PET)
# - Üç algoritmayı karşılaştırmak: RandomForest, LightGBM, XGBoost
# - SMOTE kullanımı için "data leakage-safe" tasarım:
#   * CV değerlendirmesinde SMOTE yalnızca fold-train üzerinde uygulanmalı.
#   * Bunu sağlamanın en güvenli yolu: imblearn Pipeline içinde SMOTE kullanmak.
# - Sonuçları Excel’e yazmak ve temel şekilleri üretmek:
#   * Test AUC barplot (SMOTE’lu)
#   * Algoritma bazında 3 modelin ROC eğrileri (SMOTE’lu)
#   * Algoritma bazında 3 modelin Confusion Matrix’leri (SMOTE’lu)

# =============================================================================
# 1) Kütüphaneler
# =============================================================================

import pandas as pd
# pandas: Excel okumak, sonuçları tabloya dökmek, pivot hazırlamak için kullanılır.

import numpy as np
# numpy: istatistiksel hesaplar (mean/std), array işlemleri ve interpolasyon için kullanılır.

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
# train_test_split:
# - Train/Test ayrımı yaparak bağımsız test seti oluşturur.
# - Böylece CV sonuçlarından ayrı, final performans raporlanabilir.
# StratifiedKFold:
# - CV fold’larında sınıf dağılımını korur (RCB sınıfları dengesiz olabileceği için kritik).
# cross_validate:
# - Bir modelin CV sonuçlarını birden fazla metrikle aynı anda döndürür.

from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, f1_score,
    roc_curve, auc
)
# confusion_matrix: sınıf bazında hata dağılımı.
# accuracy_score: doğruluk (dengesiz sınıflarda tek başına yeterli olmayabilir).
# roc_auc_score: multiclass OVR macro AUC hesaplamak için.
# f1_score (macro): sınıfları eşit ağırlıkla değerlendirir (dengesizlikte daha adil).
# roc_curve & auc: ROC eğrisi çizmek ve AUC’i manuel hesaplamak için.

from sklearn.preprocessing import label_binarize
# label_binarize:
# - ROC eğrileri için multiclass hedefi OVR (one-vs-rest) binary matrise çevirir.

from imblearn.over_sampling import SMOTE
# SMOTE:
# - Azınlık sınıfları sentetik örneklerle artırır.
# - Dikkat: yanlış uygulanırsa (tüm veri üzerinde) leakage yaratır.

from imblearn.pipeline import Pipeline  # IMPORTANT: SMOTE inside CV folds
# imblearn Pipeline:
# - SMOTE'un sadece fit sırasında (fold-train) uygulanmasını sağlar.
# - Böylece CV sırasında validasyon fold’u SMOTE ile "kirlenmez" → leakage-safe yaklaşım.

import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib/seaborn: barplot, ROC, confusion matrix heatmap çizimleri.

from google.colab import files
# files.upload / files.download:
# - Colab’da dosya yüklemek ve çıktıları indirmek için.

# --- optional installs (Colab) ---
# !pip -q install lightgbm xgboost
# Not:
# - Colab’da LightGBM veya XGBoost yüklü değilse kurulum gerekebilir.
# - Bu satır yorumlu, yani kullanıcı isterse aktif eder.

from sklearn.ensemble import RandomForestClassifier
# RandomForestClassifier:
# - Ağaç tabanlı bir ensemble yöntemidir, ölçekleme gerektirmez.
# - Kategorik kodlanmış sayısal verilerle iyi çalışır.

# -----------------------------------------------------------------------------
# 1.1) LightGBM opsiyonel import
# -----------------------------------------------------------------------------
try:
    from lightgbm import LGBMClassifier
    has_lgbm = True
    # has_lgbm:
    # - LightGBM başarıyla import edilirse algoritmalar listesine eklenecek.
except Exception as e:
    has_lgbm = False
    # Import edilemezse script durmasın; sadece uyarı verip devam etsin.
    print(" LightGBM import edilemedi. Kurmak için: !pip install lightgbm")

# -----------------------------------------------------------------------------
# 1.2) XGBoost opsiyonel import
# -----------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    has_xgb = True
    # has_xgb:
    # - XGBoost başarıyla import edilirse algoritmalar listesine eklenecek.
except Exception as e:
    has_xgb = False
    print(" XGBoost import edilemedi. Kurmak için: !pip install xgboost")

# =============================================================================
# 2) Görsel ayarlar
# =============================================================================

# Matplotlib varsayılan figür boyutunu büyüt:
# - Çok sınıflı ROC/CM grafiklerinde okunabilirliği artırır.
plt.rcParams['figure.figsize'] = (12, 8)

# Grafiklerde yazı boyutu:
plt.rcParams['font.size'] = 12

# Seaborn stil:
# - Beyaz zemin + grid çizgileri → tez/rapor için daha okunur.
sns.set_style("whitegrid")

# =============================================================================
# 3) Global sabitler
# =============================================================================

RANDOM_STATE = 42
# RANDOM_STATE:
# - Train/test split, CV shuffle, SMOTE random üretimi gibi işlemleri deterministik yapar.
# - Tez/GitHub için “reproducibility” sağlar.

N_SPLITS = 5
# N_SPLITS:
# - 5-fold CV: yaygın standart (bias/variance dengesi iyi).

print("=== PET VERİLERİ İLE RCB SINIFLANDIRMA (RF + LGBM + XGB) + ŞEKİLLER ===")

# =============================================================================
# 4) Veri yükleme
# =============================================================================

print("Lütfen Excel dosyanızı yükleyin:")

uploaded = files.upload()
# files.upload():
# - Colab UI üzerinden dosya seçtirir.
# - Dönen yapı bir dict (dosya_adı -> bytes) şeklindedir.

file_name = list(uploaded.keys())[0]
# Yüklenen dosyaların ilkini alır.
# (Tek dosya yüklendiği varsayımıyla pratik bir yöntem.)

data = pd.read_excel(file_name)
# Excel’den DataFrame oluşturur.
# Not: Büyük dosyalarda read_excel yavaş olabilir; ama tez verileri genelde yönetilebilir.

print(f"Veri yüklendi: {data.shape}")
# Veri boyutu kontrolü: satır x sütun

# =============================================================================
# 5) Feature listeleri
# =============================================================================

pet_features = [
    'SUVmax', 'SUVmean4', 'TLG', 'MTV', 'Yüzey/Hacim Oranı4', 'Küresellik4',
    'Asferisite4', 'SUV Varyansı4', 'SUV Eğriliği4', 'GLCM Entropi4',
    'GLCM Kontrast4', 'GLRLM Non-Uniformite4', 'NGTDM Coarseness4', 'GLSZM Entropi4'
]
# pet_features:
# - PET görüntüleme metrikleri.
# - Bu kolonların bir kısmı bazı hastalarda eksik olabilir (NaN).
# - Bir hasta "PET verili" sayılabilsin diye bu kolonların hepsinin dolu olması isteniyor (aşağıda dropna ile).

all_features = [
    'i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i12',
    'i13','i14','i15','i46','i47',
    'i16','i17','i18','i19','i45',
    'i21','i22','i23','i24','i25','i26','i27','i28','i29','i30',
    'i31','i32','i33','i34','i35','i36','i37','i38','i39','i40','i41','i42','i43','i44',
    'i48','i49','i50','i51','i52','i53','i54','i55','i56','i57','i58','i59','i60','i61','i62','i63','i64'
]
# all_features:
# - Klinik/patoloji/onkoloji/demografi/komorbidite/biyokimya/radyoloji birleşik seti.
# - i11 ve i20 zaten yok (önceki ana akış kararınla uyumlu).

target = 'RCB_Kategorize'
# Hedef kolon: çok sınıflı etiket (RCB 0/1/2/3 gibi)

# =============================================================================
# 6) PET verisi olan hastaları seçme
# =============================================================================

print(f"\n=== PET VERİSİ OLAN HASTALAR ===")

pet_data = data.dropna(subset=pet_features)
# dropna(subset=pet_features):
# - PET feature’larından herhangi biri NaN ise o satırı eler.
# - Böylece Model PET ve Model ALL+PET için “tam PET feature” şartı sağlanır.
# Kritik not:
# - Bu seçim, analizin evrenini “PET’i eksiksiz olan hastalar” ile sınırlar.
# - Dolayısıyla Model ALL sonuçları bile bu alt kohort üzerinde hesaplanacak.

print(f"PET verisi olan hasta sayısı: {len(pet_data)}")

# =============================================================================
# 7) Kategorik PET feature'larını sayısala çevirme
# =============================================================================

print(f"\n=== KATEGORİK DEĞİŞKENLERİ SAYISAL KODLARA ÇEVİRME ===")

pet_data_encoded = pet_data.copy()
# Orijinal veriyi bozmamak için kopya üzerinde çalışılır.

for col in pet_features:
    # PET feature listesinde dolaşıyoruz (hedef: PET feature’larında object varsa encode etmek)
    if pet_data_encoded[col].dtype == 'object':
        # Eğer sütun object ise (örn. metin/kategorik etiketler):
        print(f"'{col}' kategorik → sayısal kodlama")

        # category + cat.codes:
        # - her kategoriye 0..n-1 arası kod verir.
        # - Model için sayısal input üretir.
        # UYARI (tez notu):
        # - Bu kodlama sıralı anlam taşımaz; sadece ID atamasıdır.
        # - Eğer kategoriler ordinal ise farklı kodlama gerekebilir.
        pet_data_encoded[col] = pet_data_encoded[col].astype('category').cat.codes
    else:
        # Zaten sayısal ise dokunmuyoruz (float/int).
        print(f"'{col}' zaten sayısal")

# =============================================================================
# 8) Target kodlama + sınıf sayısı
# =============================================================================

pet_data_encoded[target] = pet_data_encoded[target].astype('category').cat.codes
# Target’ı kategorik olarak ele alıp 0..K-1 kodlarına çeviriyoruz.
# Not:
# - Eğer zaten 0/1/2/3 şeklindeyse bu adım genellikle aynı sonucu verir.
# - Ama veri setinde “RCB-I” gibi string etiketler varsa bunu güvenli şekilde sayısala indirger.

n_classes = pet_data_encoded[target].nunique()
print(f"\nSınıf sayısı: {n_classes} (0..{n_classes-1})")
# n_classes:
# - ROC binarize, XGBoost num_class ve confusion matrix etiketleri için kullanılacak.
# - Veri setinde bazı sınıflar yoksa (ör. hiç RCB-3 yoksa) n_classes 3 olabilir.

# =============================================================================
# BÖLÜM 2 — MODEL TANIMI + ALGORİTMALAR + CV SETUP + SONUÇ KONTEYNERLERİ
# =============================================================================
# Bu bölümün amacı:
# 1) Üç farklı feature setini (Model PET, Model ALL, Model ALL+PET) tanımlamak
#    ve karşılaştırmayı mümkün kılmak.
# 2) Kullanılacak algoritmaları tek bir sözlükte toplamak:
#    - RandomForest (her zaman var)
#    - LightGBM (varsa)
#    - XGBoost (varsa)
# 3) CV (Cross-Validation) kurulumunu ve hangi metriklerin ölçüleceğini belirlemek:
#    - AUC (multiclass OVR)
#    - Accuracy
#    - Macro F1
# 4) Çıktıları raporlamak için “results” listesi ve
#    grafik/CM/ROC üretimi için “best_test_artifacts” saklama yapısını hazırlamak.

# -----------------------------------------------------------------------------
# 9) 3 model tanımı
# -----------------------------------------------------------------------------
models = {
    'Model PET': pet_features,
    # Sadece PET feature’ları ile modelleme:
    # - PET sinyalinin tek başına RCB sınıflamasına katkısını görmek için.

    'Model ALL': all_features,
    # PET harici tüm klinik/patoloji/radyoloji özellikleri:
    # - “klasik” klinik+patoloji modelinin performansı için referans.

    'Model ALL + PET': all_features + pet_features
    # PET ile klinik özellikleri birleştiren model:
    # - PET eklenince performans artıyor mu?
    # - En önemli karşılaştırmalardan biri.
}

print(f"\n=== 3 MODEL TANIMI ===")
for model_name, feats in models.items():
    # Her modelin kaç adet feature içerdiğini loglamak:
    # - Sonuçları yorumlarken model karmaşıklığı/özellik sayısı farkını bilmek önemlidir.
    print(f"{model_name}: {len(feats)} özellik")

# -----------------------------------------------------------------------------
# 10) Algoritmalar
# -----------------------------------------------------------------------------
algorithms = {}
# algorithms sözlüğü:
# - key: algoritma adı (str)
# - value: sklearn benzeri estimator
# Böylece ana döngüde “hangi algoritmalar var?” sorusu tek yerden yönetilir.

# 10.1) RandomForest (her ortamda çalışır)
algorithms["RandomForest"] = RandomForestClassifier(
    random_state=RANDOM_STATE,
    # random_state: tekrarlanabilir sonuçlar (ağaç örnekleme rastgeleliği sabitlenir)

    n_estimators=200,
    # n_estimators: ağaç sayısı
    # - Daha fazla ağaç genelde performansı artırır ama süreyi uzatır.
    # - 200, pratik bir denge.

    max_depth=10,
    # max_depth: ağaç derinliği
    # - Aşırı derin ağaçlar overfitting’e gider.
    # - 10 gibi bir sınır genelde düzenleyici etki yapar.

    min_samples_split=5,
    # Bir düğümün bölünebilmesi için en az kaç örnek olmalı.
    # - Küçük sayılar daha kompleks model demek → overfitting riski.
    # - 5 ile biraz daha kontrollü.

    min_samples_leaf=2,
    # Yaprak düğümde minimum örnek sayısı
    # - 1 yerine 2 kullanmak çoğu zaman daha stabil / daha az overfit.

    n_jobs=-1
    # n_jobs=-1: tüm CPU çekirdeklerini kullan → eğitim hızlanır
)

# 10.2) LightGBM (opsiyonel)
if has_lgbm:
    algorithms["LightGBM"] = LGBMClassifier(
        random_state=RANDOM_STATE,
        # LightGBM de boosting olduğu için rastgelelik sabitlenir.

        n_estimators=400,
        # Boosting’de “ağaç sayısı/iterasyon”:
        # - Daha fazla iterasyon = daha ince öğrenme (learning_rate ile birlikte düşünülmeli)

        learning_rate=0.05,
        # learning_rate: her yeni ağacın katkı miktarı
        # - 0.05 daha yavaş öğrenme ama genelde daha iyi genelleme.

        num_leaves=31,
        # LightGBM’nin model kapasitesini belirleyen önemli parametre:
        # - num_leaves büyürse model güçlenir ama overfit riski artar.

        subsample=0.9,
        colsample_bytree=0.9
        # subsample / colsample_bytree:
        # - Satır ve feature altörnekleme
        # - Overfitting’i azaltabilir, genellemeyi iyileştirir.
    )

# 10.3) XGBoost (opsiyonel)
if has_xgb:
    algorithms["XGBoost"] = XGBClassifier(
        random_state=RANDOM_STATE,

        n_estimators=400,
        # boosting iterasyon sayısı

        learning_rate=0.05,
        # daha düşük lr → daha stabil öğrenme (genelde daha iyi genelleme)

        max_depth=4,
        # ağaç derinliği: 4 gibi nispeten sığ ağaçlar
        # - overfitting’i azaltmak için tercih edilir.

        subsample=0.9,
        colsample_bytree=0.9,
        # satır/feature altörnekleme → regularization etkisi

        reg_lambda=1.0,
        # L2 regularization:
        # - ağırlıkları cezalandırır, aşırı uyumu azaltır.

        objective="multi:softprob",
        # multi:softprob:
        # - multiclass problemde sınıf olasılıklarını döndürür.
        # - ROC/AUC ve calibration gibi analizlerde proba gerektiği için doğru seçim.

        num_class=n_classes,
        # XGBoost’ta multiclass için sınıf sayısını açıkça vermek gerekir.

        eval_metric="mlogloss",
        # eğitim sırasında izlenecek metrik (multiclass log loss)

        n_jobs=-1
    )

print("\n=== KULLANILACAK ALGORİTMALAR ===")
for k in algorithms.keys():
    # Hangi algoritmaların aktif olduğunu görmek:
    # - LightGBM/XGBoost import edilemediyse listede görünmeyecek.
    print("-", k)

# -----------------------------------------------------------------------------
# 11) CV setup + scoring
# -----------------------------------------------------------------------------
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)
# StratifiedKFold:
# - Her fold içinde sınıf dağılımını korur.
# - shuffle=True ile fold’lar karıştırılır, tesadüfi sıralama etkisi azalır.
# - random_state ile deterministik hale gelir.

scoring = {
    "auc": "roc_auc_ovr",
    # roc_auc_ovr:
    # - multiclass’ta one-vs-rest AUC hesaplar.
    # - sklearn scorer genelde macro averaging uygular (sınıflar eşit ağırlık).
    # - Özellikle dengesiz sınıflar için macro yaklaşımı daha adildir.

    "acc": "accuracy",
    # accuracy:
    # - Genel doğruluk, ama dengesiz sınıflarda yanıltıcı olabilir.
    # - Yine de rapora eklenmesi karşılaştırma açısından faydalıdır.

    "f1": "f1_macro"
    # f1_macro:
    # - Her sınıfın F1’ını hesaplar, sonra ortalamasını alır.
    # - Sınıf dengesizliği durumunda performansı daha doğru yansıtır.
}

# -----------------------------------------------------------------------------
# 12) Sonuçları toplama konteynerleri
# -----------------------------------------------------------------------------
results = []
# results:
# - Her model × algoritma × SMOTE durumu için
#   CV ve Test metriklerini dict olarak ekleyeceğiz.
# - En sonunda DataFrame yapıp Excel’e yazacağız.

best_test_artifacts = {}
# best_test_artifacts:
# - Şekil üretimi için gerekli ham çıktıları saklar.
# - Her model+algoritma için (SMOTE’lu koşulun) test tahminleri:
#   y_test, y_pred, y_proba ve eğitilmiş pipeline (SMOTE+Classifier).
# Neden saklıyoruz?
# - Döngü bittikten sonra ROC ve confusion matrix çizmek için tekrar eğitim yapmaya gerek kalmasın.

print(
    f"\n=== ANALİZ BAŞLIYOR (3 MODEL × {len(algorithms)} ALGORİTMA × SMOTE'lu/SMOTE'suz) ==="
)
# Bu log satırı:
# - Koşu başlamadan kaç kombinasyon çalışacağına dair kullanıcıya özet verir.
# - Colab çıktısında takip kolaylığı sağlar.

# =============================================================================
# BÖLÜM 3 — ANA DÖNGÜ (MODEL × ALGORİTMA) + LEAKAGE-SAFE SMOTE + METRİK TOPLAMA
# =============================================================================
# Bu bölümde şunlar yapılır:
# 1) Her model tanımı için (Model PET / Model ALL / Model ALL+PET) X ve y hazırlanır.
# 2) Her model için train/test split yapılır:
#    - Bu split, “final test değerlendirmesi” için ayrılır.
#    - Split önce yapılır ki test seti SMOTE veya eğitim sürecinden etkilenmesin.
# 3) Her algoritma için iki koşul çalıştırılır:
#    A) SMOTE’suz: CV + test
#    B) SMOTE’lu:  CV + test
#       * CV sırasında SMOTE pipeline içine alınır (leakage-safe)
# 4) Tüm metrikler results listesine eklenir.
# 5) Grafik üretmek için SMOTE’lu test tahminleri best_test_artifacts içine saklanır.
#
# KRİTİK NOT (clone kullanımı)
# ============================================================================
# algorithms sözlüğünde tuttuğumuz nesneler "şablon (template)" gibi davranmalıdır.
# Aynı model nesnesini tekrar tekrar fit etmek:
# - bazı algoritmalarda iç state / cache / internal booster yapıları gibi kalıntılara yol açabilir
# - bu da reproducibility ve metodolojik doğruluk açısından risk oluşturur.
#
# Bu yüzden:
# - SMOTE’suz koşul için ayrı bir clone() ile model örneği üretiriz: clf_no
# - SMOTE’lu koşul için ayrı bir clone() ile model örneği üretiriz: clf_sm
# Böylece her eğitim tamamen bağımsız instance ile yapılır.

from sklearn.base import clone  # Her eğitimde temiz model kopyası almak için

for model_name, feats in models.items():
    # models sözlüğündeki her model tanımı üzerinde dolaşırız.
    # model_name: örn. "Model PET"
    # feats: o modele ait feature listesi

    print(f"\n==============================")
    print(f"MODEL: {model_name} ({len(feats)} özellik)")
    print(f"==============================")

    # -------------------------------------------------------------------------
    # 3.1) X ve y’yi hazırlama
    # -------------------------------------------------------------------------
    X = pet_data_encoded[feats]
    # X: sadece ilgili modelin feature sütunlarını seçer.
    #
    # Not:
    # - pet_data_encoded zaten PET verisi eksiksiz hastalarla filtrelenmişti.
    # - Bu nedenle Model ALL burada "PET kohortu üzerindeki ALL" performansıdır.
    #   (yani ALL modelini tüm hasta havuzunda değil, PET alt-kümesinde test etmiş oluruz.)

    y = pet_data_encoded[target]
    # y: hedef etiket (0..n_classes-1 kodlu)

    # -------------------------------------------------------------------------
    # 3.2) Train/Test split (sabit)
    # -------------------------------------------------------------------------
    # Sabit split neden önemli?
    # - Tüm algoritmalar aynı test setinde karşılaştırılırsa kıyas adil olur.
    # - Aksi durumda her algoritma farklı test örnekleri görür ve karşılaştırma “gürültülü” olur.
    #
    # Data leakage açısından:
    # - Split önce yapılır.
    # - SMOTE sadece train tarafında yapılır.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )
    # stratify=y:
    # - Train/test içinde sınıf oranlarını korur.
    # - Özellikle azınlık sınıfların testte kaybolmasını önler.

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # -------------------------------------------------------------------------
    # 3.3) Model içindeki her algoritmayı çalıştırma
    # -------------------------------------------------------------------------
    # Burada "clf" diye doğrudan model nesnesini kullanmıyoruz.
    # Onun yerine clf_template (şablon) kullanıp her koşulda clone üretiyoruz.
    for algo_name, clf_template in algorithms.items():
        print(f"\n--- Algorithm: {algo_name} ---")

        # =====================================================================
        # A) SMOTE’suz koşul (CV + Test)
        # =====================================================================

        # ---------------------------------------------------------------------
        # 3.3.1) CV değerlendirme (SMOTE yok)
        # ---------------------------------------------------------------------
        # cross_validate:
        # - Eğitim verisini (X_train, y_train) fold’lara böler
        # - scoring sözlüğündeki metrikleri her fold’da hesaplar
        # - sonuçları "test_auc", "test_acc", "test_f1" gibi anahtarlarla döndürür
        #
        # Akademik not:
        # - cross_validate zaten kendi içinde clone yapar; ancak biz yine de
        #   template üzerinden ayrı bir clone üretip "bu koşulun modeli budur" diye
        #   açık bir state ayrımı sağlıyoruz.
        clf_no = clone(clf_template)  # SMOTE’suz koşul için bağımsız model instance

        cv_no = cross_validate(
            clf_no,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        # ---------------------------------------------------------------------
        # 3.3.2) Final eğitim (SMOTE yok) + test tahmini
        # ---------------------------------------------------------------------
        # Burada CV’den bağımsız şekilde “train’in tamamına fit” yapılır.
        # Amaç: tek bir nihai model eğitip, test setinde final performansı raporlamak.
        #
        # Önemli:
        # - Burada da clone edilmiş clf_no kullanıyoruz (template değil).
        # - Böylece bu eğitim CV içindeki modellerden de bağımsızdır.
        clf_no.fit(X_train, y_train)

        y_pred_no = clf_no.predict(X_test)
        # test setindeki sınıf tahminleri

        y_proba_no = clf_no.predict_proba(X_test)
        # test setindeki sınıf olasılıkları
        # - AUC hesaplamak için şarttır.

        # ---------------------------------------------------------------------
        # 3.3.3) Test metrikleri (SMOTE yok)
        # ---------------------------------------------------------------------
        test_auc_no = roc_auc_score(
            y_test,
            y_proba_no,
            multi_class="ovr",
            average="macro"
        )
        # multi_class="ovr":
        # - one-vs-rest AUC (çok sınıflı AUC için standart yaklaşımlardan biri)
        #
        # average="macro":
        # - her sınıfa eşit ağırlık verir (dengesiz sınıflarda daha adil)

        test_acc_no = accuracy_score(y_test, y_pred_no)
        test_f1_no = f1_score(y_test, y_pred_no, average="macro")

        # ---------------------------------------------------------------------
        # 3.3.4) Sonuç kaydı (SMOTE yok)
        # ---------------------------------------------------------------------
        results.append({
            "Model": model_name,
            "Algorithm": algo_name,
            "SMOTE_Durumu": "SMOTE'suz",

            # CV metrikleri:
            "CV_AUC_Mean": np.mean(cv_no["test_auc"]),
            "CV_AUC_Std":  np.std(cv_no["test_auc"], ddof=0),

            "CV_Accuracy_Mean": np.mean(cv_no["test_acc"]),
            "CV_Accuracy_Std":  np.std(cv_no["test_acc"], ddof=0),

            "CV_F1_Mean": np.mean(cv_no["test_f1"]),
            "CV_F1_Std":  np.std(cv_no["test_f1"], ddof=0),

            # Test metrikleri:
            "Test_AUC": test_auc_no,
            "Test_Accuracy": test_acc_no,
            "Test_F1": test_f1_no
        })

        # =====================================================================
        # B) SMOTE’lu koşul (CV + Test)  [LEAKAGE-SAFE]
        # =====================================================================
        # Buradaki en kritik nokta:
        #
        # Eğer SMOTE’u CV’den ÖNCE tüm X_train üzerinde uygularsan:
        # - fold-val örnekleri de SMOTE üretimine “dolaylı” olarak karışır
        # - model validasyon verisinden bilgi sızdırmış olur
        # - CV skorları yapay şekilde şişer → data leakage
        #
        # Bu yüzden SMOTE mutlaka Pipeline içinde olmalı:
        # - CV fold’larında fit çağrıldığında:
        #   SMOTE yalnızca fold-train üzerine uygulanır.
        # - fold-val kısmı SMOTE görmez.

        # SMOTE’lu koşul için de bağımsız model instance üret:
        clf_sm = clone(clf_template)

        smote = SMOTE(random_state=RANDOM_STATE)
        # random_state:
        # - sentetik örnek üretimi her çalıştırmada aynı olsun diye

        pipe = Pipeline([
            ("smote", smote),
            ("clf", clf_sm)
        ])
        # Pipeline sırası:
        # 1) smote.fit_resample(fold_train)
        # 2) clf.fit(resampled_fold_train)
        #
        # NOT:
        # - Validation fold’u SMOTE görmez → leakage-safe.

        # ---------------------------------------------------------------------
        # 3.3.5) CV değerlendirme (SMOTE var, pipeline içinde)
        # ---------------------------------------------------------------------
        cv_yes = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        # ---------------------------------------------------------------------
        # 3.3.6) Final eğitim (SMOTE var) + test tahmini
        # ---------------------------------------------------------------------
        # pipe.fit(X_train, y_train):
        # - X_train üzerinde SMOTE uygular
        # - sonra classifier’ı SMOTE sonrası veriyle eğitir
        #
        # Test setine dokunulmaz → final değerlendirme güvenli kalır.
        pipe.fit(X_train, y_train)

        y_pred_yes = pipe.predict(X_test)
        y_proba_yes = pipe.predict_proba(X_test)

        # ---------------------------------------------------------------------
        # 3.3.7) Test metrikleri (SMOTE var)
        # ---------------------------------------------------------------------
        test_auc_yes = roc_auc_score(
            y_test,
            y_proba_yes,
            multi_class="ovr",
            average="macro"
        )
        test_acc_yes = accuracy_score(y_test, y_pred_yes)
        test_f1_yes = f1_score(y_test, y_pred_yes, average="macro")

        # ---------------------------------------------------------------------
        # 3.3.8) Sonuç kaydı (SMOTE var)
        # ---------------------------------------------------------------------
        results.append({
            "Model": model_name,
            "Algorithm": algo_name,
            "SMOTE_Durumu": "SMOTE'lu",

            # CV metrikleri:
            "CV_AUC_Mean": np.mean(cv_yes["test_auc"]),
            "CV_AUC_Std":  np.std(cv_yes["test_auc"], ddof=0),

            "CV_Accuracy_Mean": np.mean(cv_yes["test_acc"]),
            "CV_Accuracy_Std":  np.std(cv_yes["test_acc"], ddof=0),

            "CV_F1_Mean": np.mean(cv_yes["test_f1"]),
            "CV_F1_Std":  np.std(cv_yes["test_f1"], ddof=0),

            # Test metrikleri:
            "Test_AUC": test_auc_yes,
            "Test_Accuracy": test_acc_yes,
            "Test_F1": test_f1_yes
        })

        # ---------------------------------------------------------------------
        # 3.3.9) Şekiller için artefact saklama (SMOTE’lu koşul)
        # ---------------------------------------------------------------------
        # Tez raporlamasında çoğunlukla SMOTE’lu durumda:
        # - ROC eğrileri
        # - Confusion matrix
        # daha “sınıf adaleti” yüksek olduğu için tercih edilir.
        #
        # Bu nedenle görselleştirme için SMOTE’lu test tahminlerini saklıyoruz.
        best_test_artifacts[(model_name, algo_name)] = {
            "y_test": y_test,
            "y_pred": y_pred_yes,
            "y_proba": y_proba_yes,
            "fitted": pipe
            # fitted:
            # - SMOTE + classifier pipeline’ı
            # - Daha sonra istenirse feature importance gibi analizlerde kullanılabilir.
        }

        # ---------------------------------------------------------------------
        # 3.3.10) Konsola özet yazdırma
        # ---------------------------------------------------------------------
        print(
            f"SMOTE'suz  | CV AUC: {np.mean(cv_no['test_auc']):.3f}±{np.std(cv_no['test_auc']):.3f} "
            f"| Test AUC: {test_auc_no:.3f}"
        )
        print(
            f"SMOTE'lu   | CV AUC: {np.mean(cv_yes['test_auc']):.3f}±{np.std(cv_yes['test_auc']):.3f} "
            f"| Test AUC: {test_auc_yes:.3f}"
        )

# =============================================================================
# BÖLÜM 4 — SONUÇ TABLOSU + EXCEL RAPORU + ŞEKİLLER + DOSYA İNDİRME
# =============================================================================
# Bu bölümde amaç:
# 1) results listesindeki tüm satırları DataFrame'e çevirip tek bir özet tablo oluşturmak.
# 2) Bu tabloyu Excel’e yazmak:
#    - Results_Long: "uzun format" (her model×algoritma×SMOTE bir satır)
#    - Results_Pivot: hızlı kıyas için pivot özet
# 3) Şekilleri üretmek:
#    - Test AUC barplot (SMOTE’lu koşul)
#    - ROC eğrileri: her algoritma için 3 modeli aynı grafikte (SMOTE’lu)
#    - Confusion matrix: her algoritma için 3 modeli yan yana (SMOTE’lu)
# 4) Üretilen dosyaları Colab üzerinden indirmek.

# -----------------------------------------------------------------------------
# 4.1) Sonuç tablosu (long format)
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
# results_df:
# - Ana döngüde results listesine eklediğimiz dict’leri tabloya çevirir.

print("\n=== TÜM SONUÇLAR (ÖZET) ===")
print(
    results_df
    .sort_values(["Model", "Algorithm", "SMOTE_Durumu"])
    .to_string(index=False)
)
# sort_values:
# - Çıktının konsolda okunabilir olması için sonuçları düzenler.
# to_string(index=False):
# - DataFrame’i konsola düzgün formatta basar (index göstermeden).

# -----------------------------------------------------------------------------
# 4.2) Excel’e kaydetme (long + pivot)
# -----------------------------------------------------------------------------
# Excel çıktısı neden önemli?
# - Tez eklerinde tabloları kolay üretmek
# - Reviewer/jüriye "ham sonuçları" şeffaf sunmak
# - Pivot ile hızlı karşılaştırma yapmak

with pd.ExcelWriter("PET_Analiz_Sonuclari_TUM_ALG.xlsx", engine="openpyxl") as writer:
    # Results_Long:
    # - Her koşul (SMOTE’lu/SMOTE’suz) ayrı satırdır.
    # - Bu format, istatistiksel analiz veya farklı görselleştirme araçları için idealdir.
    results_df.to_excel(writer, sheet_name="Results_Long", index=False)

    # Pivot neden?
    # - Aynı model+algoritma için SMOTE’lu ve SMOTE’suz değerleri yan yana görmek kolaylaşır.
    # - Tek bakışta “SMOTE etkisi” okunabilir.
    pivot_auc = results_df.pivot_table(
        index=["Model", "Algorithm"],
        columns="SMOTE_Durumu",
        values=[
            "CV_AUC_Mean", "CV_AUC_Std",
            "Test_AUC", "Test_Accuracy", "Test_F1"
        ],
        aggfunc="first"
        # aggfunc="first":
        # - Her index/column kombinasyonu tek değer içerdiği için ilkini alıyoruz.
        # - Ortalama alma gerekmiyor çünkü zaten bir satır var.
    )

    pivot_auc.to_excel(writer, sheet_name="Results_Pivot")

print("\n Sonuçlar 'PET_Analiz_Sonuclari_TUM_ALG.xlsx' dosyasına kaydedildi!")

# -----------------------------------------------------------------------------
# 4.3) ŞEKİLLER
# -----------------------------------------------------------------------------
print("\n=== ŞEKİLLER ÜRETİLİYOR ===")

# -----------------------------------------------------------------------------
# 4.3.1) Test AUC barplot (SMOTE'lu)
# -----------------------------------------------------------------------------
# Neden yalnız SMOTE’lu?
# - SMOTE’lu koşul genellikle azınlık sınıflara daha adil performans verir.
# - Tezde karşılaştırma grafikleri için daha anlamlı olabilir.
#
# Not:
# - İstersen hem SMOTE’lu hem SMOTE’suz için iki ayrı barplot üretmek de mümkün.

plt.figure(figsize=(14, 6))

smote_yes_df = results_df[results_df["SMOTE_Durumu"] == "SMOTE'lu"].copy()
# smote_yes_df:
# - yalnız SMOTE’lu satırları seçiyoruz.

sns.barplot(
    data=smote_yes_df,
    x="Model",
    y="Test_AUC",
    hue="Algorithm"
)
# barplot:
# - x ekseni: Model PET / Model ALL / Model ALL+PET
# - hue: algoritma
# - y ekseni: Test AUC

plt.title("Test AUC Karşılaştırması (SMOTE'lu) - Tüm Algoritmalar")
plt.ylim(0, 1)
# AUC 0..1 aralığında olduğu için y limitini sabitlemek görsel kıyas için iyi.

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("PET_TestAUC_SMOTEli_TUM_ALG.png", dpi=300, bbox_inches="tight")
# dpi=300:
# - tez/dergi için baskı kalitesinde çıktı

plt.show()

# -----------------------------------------------------------------------------
# 4.3.2) ROC eğrileri (SMOTE’lu test) — her algoritma için 3 modeli aynı grafikte
# -----------------------------------------------------------------------------
# Burada kullanılan fikir:
# - Multiclass ROC’u sınıf sınıf çizmek çok kalabalık olur.
# - Bu yüzden her model için “macro-average ROC” üretip tek eğri ile gösteriyoruz.
#
# Hesap yöntemi:
# 1) label_binarize ile y_test OVR matrise çevrilir.
# 2) Her sınıf için ROC hesaplanır.
# 3) FPR değerleri birleştirilip mean_tpr interpolasyonla hesaplanır.
# 4) Macro ROC eğrisi ve AUC bulunur.

colors = ["blue", "red", "green"]
# 3 model olduğu için 3 renk seçildi (Model PET / ALL / ALL+PET sırasıyla)

model_order = list(models.keys())
# model_order:
# - models sözlüğünün sıradaki model isimlerini alır.
# - Böylece ROC ve CM grafiklerinde tutarlı sıralama yapılır.

for algo_name in algorithms.keys():
    # Her algoritma için ayrı ROC figürü üretiriz.
    plt.figure(figsize=(12, 8))

    for i, mname in enumerate(model_order):
        # Her algoritma içinde 3 modeli aynı grafiğe çizmek için döngü

        art = best_test_artifacts[(mname, algo_name)]
        # best_test_artifacts:
        # - ana döngüde sakladığımız SMOTE’lu test tahminlerini içerir.
        # - Burada tekrar eğitim yapmayız → daha hızlı ve tutarlı.

        y_test = art["y_test"]
        y_proba = art["y_proba"]

        classes = list(range(n_classes))
        # classes:
        # - ROC için OVR sınıf indeksleri (0..n_classes-1)

        y_bin = label_binarize(y_test, classes=classes)
        # y_bin: (n_samples, n_classes)
        # - Her satır: gerçek sınıfı 1, diğerlerini 0 yapan OVR kodlama

        fpr, tpr = {}, {}
        # Her sınıf için ROC eğrisi saklamak için dict

        for c in classes:
            # ROC eğrisi her sınıf için ayrı hesaplanır:
            fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], y_proba[:, c])

        # Macro-average ROC için tüm fpr noktalarını tek set yap:
        all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))

        mean_tpr = np.zeros_like(all_fpr)
        # mean_tpr:
        # - tüm sınıfların TPR değerleri ortak fpr gridinde interpolasyonla toplanır

        for c in classes:
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
            # np.interp:
            # - sınıf ROC eğrisini ortak fpr eksenine taşır

        mean_tpr /= len(classes)
        # sınıf sayısına böl → macro ortalama

        mean_auc = auc(all_fpr, mean_tpr)
        # macro-average eğri altında alan

        plt.plot(
            all_fpr,
            mean_tpr,
            color=colors[i],
            lw=2,
            label=f"{mname} (macro AUC={mean_auc:.3f})"
        )

    # Referans diagonal (rastgele sınıflandırıcı)
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Eğrileri (SMOTE'lu Test) - {algo_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"PET_ROC_SMOTEli_{algo_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

# -----------------------------------------------------------------------------
# 4.3.3) Confusion matrix (SMOTE’lu) — her algoritma için 3 model yan yana
# -----------------------------------------------------------------------------
# Bu grafik:
# - Her algoritma için tek bir satırda 3 farklı modelin confusion matrix’ini gösterir.
# - Böylece “PET eklenince hangi sınıflar daha iyi?” görsel olarak anlaşılır.

for algo_name in algorithms.keys():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # 1x3 grid:
    # - 3 model için yan yana panel

    for i, mname in enumerate(model_order):
        art = best_test_artifacts[(mname, algo_name)]
        y_test = art["y_test"]
        y_pred = art["y_pred"]

        cm = confusion_matrix(y_test, y_pred)
        # cm:
        # - satırlar gerçek sınıf
        # - sütunlar tahmin sınıf

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["RCB-0", "RCB-I", "RCB-II", "RCB-III"][:n_classes],
            yticklabels=["RCB-0", "RCB-I", "RCB-II", "RCB-III"][:n_classes]
        )
        # xticklabels/yticklabels:
        # - n_classes 4 değilse slice ile uyum sağlanır (ör. 3 sınıf varsa 3 etiket göster)
        # - Tezde okunabilirlik için Roman numeral format tercih edilmiştir.

        axes[i].set_title(f"{mname}\n{algo_name} (SMOTE'lu)")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"PET_CM_SMOTEli_{algo_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

# -----------------------------------------------------------------------------
# 4.4) Üretilen dosyaları konsola özetle
# -----------------------------------------------------------------------------
print("\n Şekiller üretildi:")
print("- PET_TestAUC_SMOTEli_TUM_ALG.png")
for algo_name in algorithms.keys():
    print(f"- PET_ROC_SMOTEli_{algo_name}.png")
    print(f"- PET_CM_SMOTEli_{algo_name}.png")

# -----------------------------------------------------------------------------
# 4.5) Dosyaları indir (Colab)
# -----------------------------------------------------------------------------
# Colab ortamında kullanıcıya dosyaları otomatik indirir.
# GitHub’da bu satırlar gerekmeyebilir; ama Colab notebook akışında pratiktir.

files.download("PET_Analiz_Sonuclari_TUM_ALG.xlsx")
files.download("PET_TestAUC_SMOTEli_TUM_ALG.png")

for algo_name in algorithms.keys():
    files.download(f"PET_ROC_SMOTEli_{algo_name}.png")
    files.download(f"PET_CM_SMOTEli_{algo_name}.png")

print("\n📁 Dosyalar indirildi.")


