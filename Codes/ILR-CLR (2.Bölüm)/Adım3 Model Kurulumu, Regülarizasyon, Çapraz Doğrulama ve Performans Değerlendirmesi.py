# ============================================================================
# ÇOKLU LOJİSTİK REGRESYON ANALİZİ - ADIM 3: MODEL KURULUMU VE DEĞERLENDİRME
# ============================================================================
# Bu kod:
# 1) ADIM 2 çıktılarıyla (VIF + dummy info) aynı dummy matrisi yeniden üretir
# 2) Perfect multicollinearity (VIF >= 999) olan dummy değişkenleri çıkarır
# 3) Yüksek VIF’li dummy değişkenler arasında |r| > eşik olan çiftlerde, VIF’i yüksek olanı çıkarır
# 4) Train-test split yapar (80/20, stratifiye)
# 5) 5-fold Stratified CV yapar
# 6) L1 (Lasso) ve L2 (Ridge) regularization ile Logistic Regression kurar
# 7) ROC-AUC, Accuracy, Sensitivity, Specificity, PPV, NPV hesaplar
# 8) Overfitting kontrolü yapar (Train AUC - Test AUC)
# 9) Youden’s J ile optimal eşik bulur (train üzerinde) ve testte değerlendirir
# 10) Sonuçları JSON/Excel olarak kaydeder
#
# NOT:
# - Bu sürüm SADECE Google Colab ortamında çalışacak şekilde üretilmiştir.
# ============================================================================

# ============================================================================
# 0) KÜTÜPHANELER
# ============================================================================

import json  # ADIM 2 çıktılarını (JSON) okumak ve sonuçları JSON olarak kaydetmek için.
import numpy as np  # Sayısal hesaplar (ortalama, std, exp, argmax vb.) için.
import pandas as pd  # Excel okuma ve veri çerçevesi (DataFrame) işlemleri için.

import warnings  # Uyarı mesajlarını yönetmek için.
warnings.filterwarnings("ignore")  # Eğitim/fit sırasında çıkan gereksiz uyarıları bastırmak için.

import statsmodels.api as sm  # Modelde kullanılmıyor ileride statsmodels ile kurmak için hazır

from sklearn.model_selection import train_test_split  # Veriyi train/test olarak bölmek için.
from sklearn.model_selection import StratifiedKFold  # Stratifiye cross-validation katlama için.
from sklearn.model_selection import GridSearchCV  # En iyi C parametresini aramak için.

from sklearn.linear_model import LogisticRegression  # L1/L2 regularization'lı lojistik regresyon modeli için.

from sklearn.metrics import roc_auc_score  # ROC-AUC hesaplamak için.
from sklearn.metrics import confusion_matrix  # Confusion matrix (TN/FP/FN/TP) çıkarmak için.
from sklearn.metrics import accuracy_score  # Accuracy hesaplamak için.
from sklearn.metrics import roc_curve  # ROC eğrisi + threshold listesi üretmek için.

from google.colab import files  # Colab dosya yükleme/indirme arayüzü için.

# ============================================================================
# 1) BAŞLIK / BAŞLANGIÇ BİLGİSİ
# ============================================================================

print("=" * 80)  # Konsolda görsel ayraç çizgisi (okunabilirlik).
print("ÇOKLU LOJİSTİK REGRESYON - ADIM 3: MODEL KURULUMU VE DEĞERLENDİRME")  # Başlık satırı.
print("=" * 80)  # Konsolda görsel ayraç çizgisi.
print()  # Boş satır 

# ============================================================================
# 2) VERİ YÜKLEME (COLAB)
# ============================================================================

print("1. VERİ YÜKLEME")  # Bölüm başlığı yazdır.
print("-" * 80)  # Bölüm alt çizgisi (okunabilirlik).

print("Lütfen Excel dosyanızı yükleyin:")  # Kullanıcıya Colab upload isteği göster.
uploaded = files.upload()  # Colab arayüzünden dosya yüklet (dict olarak döner).
excel_path = list(uploaded.keys())[0]  # Yüklenen ilk dosyanın adını al

data = pd.read_excel(excel_path)  # Excel dosyasını pandas ile DataFrame’e oku.
print(f"Veri yüklendi: {data.shape[0]} hasta, {data.shape[1]} sütun\n")  # Veri boyutunu yazdır.

# ADIM 2 çıktıları yüklenir (VIF özeti ve dummy kodlama bilgileri)
try:  # Dosyalar var mı diye dene.
    with open("vif_summary_report.json", "r", encoding="utf-8") as f:  # VIF özet raporunu aç.
        vif_summary = json.load(f)  # JSON’u Python sözlüğüne çevir.
    with open("dummy_coding_info.json", "r", encoding="utf-8") as f:  # Dummy bilgilerini aç.
        dummy_info = json.load(f)  # JSON’u Python sözlüğüne çevir.
    print("ADIM 2 sonuçları yüklendi (vif_summary_report.json + dummy_coding_info.json)\n")  # Başarılı mesajı.
except FileNotFoundError:  # Dosyalar yoksa buraya düşer.
    print(" HATA: ADIM 2 sonuçları bulunamadı!")  # Hata mesajı.
    print("   Lütfen önce ADIM 2'yi çalıştırın ve çıktı dosyalarının aynı çalışma dizininde olduğundan emin olun.")  # Ne yapılmalı.
    raise  # Programı durdur (aksi halde sonraki adımlar hatalı olur).

# ADIM 2’nin detay CSV VIF çıktıları (her dummy kolon için VIF değerleri)
vif1 = pd.read_csv("model1_vif_results.csv")  # Model 1 VIF sonuçlarını oku.
vif2 = pd.read_csv("model2_vif_results.csv")  # Model 2 VIF sonuçlarını oku.

# ============================================================================
# 3) TARGET (BAĞIMLI DEĞİŞKEN) KONTROLÜ
# ============================================================================

target = "RCB_ML"  # Bağımlı değişken adı (0/1 pCR gibi).

if target not in data.columns:  # Veri setinde target sütunu var mı kontrol et.
    raise ValueError(f"HATA: '{target}' sütunu veri setinde yok!")  # Yoksa anlamlı hata üret.

y = data[target].astype(int)  # Target sütununu int'e çevir (0/1 beklenir).

if y.nunique() < 2:  # Lojistik regresyon için en az 2 sınıf olmalı.
    raise ValueError(" HATA: RCB_ML tek sınıf içeriyor. Lojistik regresyon için 0 ve 1 olmalı!")  # Tek sınıfsa durdur.

print(f"RCB_ML sınıf dağılımı:\n{y.value_counts()}\n")  # Sınıf dağılımını yazdır (dengeyi görmek için).

# ============================================================================
# 4) DUMMY DEĞİŞKENLERİ OLUŞTURMA (ADIM 2 İLE UYUMLU)
# ============================================================================

print("2. DUMMY DEĞİŞKENLERİ OLUŞTURMA")  # Bölüm başlığı yazdır.
print("-" * 80)  # Bölüm alt çizgisi.

def create_dummy_variables(df_in, var_dict, ref_dict):  # Dummy üretim fonksiyonu tanımla.
    """
    ADIM 2 ile aynı mantıkla kategorik değişkenleri dummy (one-hot) kodlar.
    - Referans kategori çıkarılır (dummy trap önlemek için).
    - Kolon isimleri i1_cat2 gibi okunur hale getirilir.
    - Eksik kategoriler için 0 sütunu eklenir (güvence).
    """
    X_dummy = pd.DataFrame(index=df_in.index)  # Çıktı DataFrame’ini aynı index ile başlat (satır uyumu).

    for var_code, var_name in var_dict.items():  # Her değişken kodu için dolaş.
        if var_code not in df_in.columns:  # Veri setinde yoksa
            continue  # bu değişkeni atla.

        categories = sorted(df_in[var_code].dropna().unique())  # NaN hariç kategorileri al ve sırala.
        if len(categories) == 0:  # Kategori yoksa (tamamen boşsa)
            continue  # atla.

        ref_cat = ref_dict.get(var_code, categories[0])  # Referans kategori: ADIM 2'den gelen veya ilk kategori.

        dummy_cols = pd.get_dummies(  # One-hot encoding yap.
            df_in[var_code],  # İlgili sütun.
            prefix=var_code,  # Kolon isimleri var_code ile başlasın (i1_... gibi).
            drop_first=False  # İlkini otomatik düşürme 

        ).astype(int)  # 0/1'leri int’e çevir (VIF / model için net).

        ref_col = f"{var_code}_{ref_cat}"  # Referans dummy kolon adı (ör: i1_2).
        if ref_col in dummy_cols.columns:  # Referans kolonu varsa
            dummy_cols = dummy_cols.drop(columns=[ref_col])  # referans kolonunu düşür.

        rename_map = {}  # Yeni kolon adları için sözlük hazırla.
        for col in dummy_cols.columns:  # Üretilen her dummy kolonu dolaş.
            parts = col.split("_", 1)  # "i1_3" -> ["i1","3"] gibi.
            cat_part = parts[1] if len(parts) > 1 else "NA"  # Güvenli şekilde kategori kısmını al.
            rename_map[col] = f"{var_code}_cat{cat_part}"  # i1_3 -> i1_cat3 dönüştür.
        dummy_cols = dummy_cols.rename(columns=rename_map)  # Yeniden adlandırmayı uygula.

        for cat in categories:  # Var olan tüm kategorileri dolaş.
            if cat == ref_cat:  # Referans kategoriyse
                continue  # referans kolon zaten yok (düşürüldü).
            col_name = f"{var_code}_cat{cat}"  # Beklenen dummy kolon adı.
            if col_name not in dummy_cols.columns:  # Eğer bazı sebeple oluşmamışsa
                dummy_cols[col_name] = 0  # 0 sütunu ekle (model matrisi tutarlı kalsın).

        dummy_cols = dummy_cols[sorted(dummy_cols.columns)]  # Kolonları alfabetik sırala (düzenli görünüm).

        X_dummy = pd.concat([X_dummy, dummy_cols], axis=1)  # Bu değişkenin dummy kolonlarını ana matrise ekle.

    return X_dummy.astype(float)  # Model için float’a çevir (sklearn güvenli çalışır).

# Model değişken sözlüklerini dummy_info’dan üret
model1_vars_dict = {k: v["name"] for k, v in dummy_info["model1"].items()}  # Model1: {i1:'...', i2:'...'}.
model2_vars_dict = {k: v["name"] for k, v in dummy_info["model2"].items()}  # Model2: {i1:'...', ...}.
reference_categories = dummy_info["reference_categories"]  # ADIM2’de seçilen referans kategoriler.

X1_dummy = create_dummy_variables(data, model1_vars_dict, reference_categories)  # Model1 dummy matrisi üret.
X2_dummy = create_dummy_variables(data, model2_vars_dict, reference_categories)  # Model2 dummy matrisi üret.

print(f"Model 1: {X1_dummy.shape[1]} dummy değişken oluşturuldu")  # Model1 dummy kolon sayısı.
print(f"Model 2: {X2_dummy.shape[1]} dummy değişken oluşturuldu\n")  # Model2 dummy kolon sayısı.

# ============================================================================
# 5) MULTICOLLINEARITY DÜZELTMESİ
# ============================================================================
# Strateji:
# - VIF >= 999 olanları (perfect multicollinearity) direkt çıkar
# - VIF>=5 olanlar arasında |r| > 0.7 çiftleri bul, VIF’i daha yüksek olanı çıkar
# ============================================================================

print("3. MULTICOLLINEARITY DÜZELTMESİ")  # Bölüm başlığı.
print("-" * 80)  # Bölüm alt çizgisi.

def remove_multicollinearity(X_dummy, vif_df, correlation_threshold=0.7, vif_threshold=5.0):  # Düzeltme fonksiyonu.
    """
    1) VIF>=999 kolonları (perfect multicollinearity) çıkarır.
    2) VIF>=vif_threshold olan kolonlar içinde |r|>threshold çiftlerinde,
       VIF’i daha yüksek olan kolonu çıkarır.
    """
    X_clean = X_dummy.copy()  # Orijinali bozmamak için kopya al.

    perfect_cols = vif_df.loc[vif_df["VIF"] >= 999.0, "Değişken"].tolist()  # VIF>=999 kolon isimleri.
    perfect_cols = [c for c in perfect_cols if c in X_clean.columns]  # Sadece matriste olanları filtrele.

    if len(perfect_cols) > 0:  # Perfect kolon var mı?
        print(f"Perfect multicollinearity (VIF>=999): {len(perfect_cols)} kolon çıkarılıyor")  # Bilgilendir.
        for c in perfect_cols:  # Tek tek yazdır.
            print(f"   - {c}")  # Kolon adı.
        X_clean = X_clean.drop(columns=perfect_cols)  # Tüm perfect kolonları düşür.
    else:
        print(" Perfect multicollinearity yok (VIF>=999 bulunmadı)")  # Yoksa mesaj ver.

    high_vif_cols = vif_df.loc[  # Yüksek VIF kolonları seç.
        (vif_df["VIF"] >= vif_threshold) & (vif_df["VIF"] < 999.0),  # Aralık: [vif_threshold, 999).
        "Değişken"  # Kolon isimleri.
    ].tolist()  # Listeye çevir.
    high_vif_cols = [c for c in high_vif_cols if c in X_clean.columns]  # Temiz matriste olanları filtrele.

    removed_due_to_corr = []  # Korelasyon nedeniyle çıkarılan kolonları kaydetmek için.

    if len(high_vif_cols) > 1:  # Korelasyon için en az 2 kolon olmalı.
        corr_matrix = X_clean[high_vif_cols].corr()  # Bu kolonların korelasyon matrisi.

        for i in range(len(corr_matrix.columns)):  # Üst üçgen taraması için i.
            for j in range(i + 1, len(corr_matrix.columns)):  # j > i olacak şekilde dolaş.
                v1 = corr_matrix.columns[i]  # 1. değişken adı.
                v2 = corr_matrix.columns[j]  # 2. değişken adı.
                r = corr_matrix.iloc[i, j]  # Korelasyon değeri.

                if abs(r) > correlation_threshold:  # Eşik üstündeyse (yüksek korelasyon)
                    vif1_val = float(vif_df.loc[vif_df["Değişken"] == v1, "VIF"].values[0])  # v1 VIF değeri.
                    vif2_val = float(vif_df.loc[vif_df["Değişken"] == v2, "VIF"].values[0])  # v2 VIF değeri.

                    var_to_remove = v1 if vif1_val >= vif2_val else v2  # VIF’i daha yüksek olanı seç.

                    if (var_to_remove in X_clean.columns) and (var_to_remove not in removed_due_to_corr):  # Daha önce çıkarılmadıysa
                        removed_due_to_corr.append(var_to_remove)  # Listeye ekle.
                        print(f"  |r|>{correlation_threshold} (r={abs(r):.3f}) → {var_to_remove} çıkarılıyor (VIF daha yüksek)")  # Gerekçe yaz.
                        X_clean = X_clean.drop(columns=[var_to_remove])  # Seçilen kolonu düşür.

        if len(removed_due_to_corr) == 0:  # Hiç çıkarma olmadıysa
            print(f" Yüksek korelasyonlu (|r|>{correlation_threshold}) çift bulunmadı")  # Bilgilendir.
    else:
        print("  Korelasyon taraması için yeterli yüksek-VIF değişken yok")  # 0/1 kolon varsa.

    return X_clean  # Temizlenmiş matrisi döndür.

print("Model 1 için düzeltme:")  # Model 1 düzeltme başlığı.
X1_clean = remove_multicollinearity(X1_dummy, vif1, correlation_threshold=0.7, vif_threshold=5.0)  # Model1 temizle.
print(f" Model 1: {X1_clean.shape[1]} değişken kaldı\n")  # Kalan kolon sayısı.

print("Model 2 için düzeltme:")  # Model 2 düzeltme başlığı.
X2_clean = remove_multicollinearity(X2_dummy, vif2, correlation_threshold=0.7, vif_threshold=5.0)  # Model2 temizle.
print(f" Model 2: {X2_clean.shape[1]} değişken kaldı\n")  # Kalan kolon sayısı.

# ============================================================================
# 6) TRAIN-TEST SPLIT (80/20, STRATIFY)
# ============================================================================

print("4. TRAIN-TEST SPLIT")  # Bölüm başlığı.
print("-" * 80)  # Bölüm alt çizgisi.

X1_train, X1_test, y1_train, y1_test = train_test_split(  # Model1 train/test ayır.
    X1_clean,  # Bağımsız değişken matrisi (temizlenmiş).
    y,  # Bağımlı değişken.
    test_size=0.2,  # %20 test.
    random_state=42,  # Tekrarlanabilirlik için sabit seed.
    stratify=y  # Sınıf oranlarını train/test’te korumak için.
)

X2_train, X2_test, y2_train, y2_test = train_test_split(  # Model2 train/test ayır.
    X2_clean,  # Bağımsız değişken matrisi (temizlenmiş).
    y,  # Bağımlı değişken.
    test_size=0.2,  # %20 test.
    random_state=42,  # Tekrarlanabilirlik.
    stratify=y  # Sınıf oranları korunsun.
)

print("Model 1:")  # Model1 bilgi başlığı.
print(f"  Train: {X1_train.shape[0]} hasta, {X1_train.shape[1]} değişken")  # Train boyutu.
print(f"  Test : {X1_test.shape[0]} hasta, {X1_test.shape[1]} değişken")  # Test boyutu.
print(f"  Train pCR oranı: {y1_train.mean():.3f}")  # Train pozitif oranı.
print(f"  Test  pCR oranı: {y1_test.mean():.3f}\n")  # Test pozitif oranı.

print("Model 2:")  # Model2 bilgi başlığı.
print(f"  Train: {X2_train.shape[0]} hasta, {X2_train.shape[1]} değişken")  # Train boyutu.
print(f"  Test : {X2_test.shape[0]} hasta, {X2_test.shape[1]} değişken")  # Test boyutu.
print(f"  Train pCR oranı: {y2_train.mean():.3f}")  # Train pozitif oranı.
print(f"  Test  pCR oranı: {y2_test.mean():.3f}\n")  # Test pozitif oranı.

# ============================================================================
# 7) MODEL EĞİTİMİ + CV + EŞİK OPTİMİZASYONU
# ============================================================================

print("5. MODEL KURULUMU VE DEĞERLENDİRME")  # Bölüm başlığı.
print("-" * 80)  # Bölüm alt çizgisi.

def evaluate_model(  # Model kurma ve değerlendirme fonksiyonu.
    X_train, X_test, y_train, y_test,  # Train/test verileri.
    model_name, regularization,  # Model adı ve regularization türü ('l1'/'l2').
    C=None,  # C verilmezse GridSearch ile bulunur.
    cv_splits=5,  # Cross-validation fold sayısı.
    correlation_threshold=0.7  # (Bu fonksiyonda kullanılmıyor; dış tasarımdan kalabilir.)
):
    """
    sklearn LogisticRegression ile:
    - L1 veya L2 regularization uygular
    - C yoksa GridSearchCV ile en iyi C’yi bulur
    - Testte ROC-AUC + confusion matrix metriklerini çıkarır
    - 5-fold CV AUC hesaplar
    - Train/Test AUC farkından overfitting (gap) ölçer
    - Youden’s J ile train üzerinde en iyi threshold bulur, testte uygular
    """

    # ------------------------------------------------------------------------
    # 7.1) Penalty/Solver seçimi
    # ------------------------------------------------------------------------
    if regularization == "l1":  # L1 seçildiyse
        penalty = "l1"  # L1 penalty.
        solver = "liblinear"  # L1’i destekleyen solver.
    elif regularization == "l2":  # L2 seçildiyse
        penalty = "l2"  # L2 penalty.
        solver = "lbfgs"  # L2 için uygun solver.
    else:
        raise ValueError("regularization sadece 'l1' veya 'l2' olabilir")  # Hatalı giriş.

    # ------------------------------------------------------------------------
    # 7.2) En iyi C seçimi (GridSearchCV) - C yoksa otomatik ara
    # ------------------------------------------------------------------------
    best_C = C  # Eğer kullanıcı C verirse onu kullanacağız.

    if best_C is None:  # C verilmemişse
        param_grid = {  # Denenecek C değerleri listesi.
            "C": [0.001, 0.01, 0.05, 0.07, 0.1, 0.144, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)  # Stratifiye CV tanımla.

        grid = GridSearchCV(  # GridSearch nesnesi oluştur.
            LogisticRegression(  # Temel model.
                penalty=penalty,  # L1/L2.
                solver=solver,  # Uygun solver.
                max_iter=2000,  # Yakınsama için iterasyon.
                random_state=42  # Tekrarlanabilirlik.
            ),
            param_grid=param_grid,  # Denenecek parametreler.
            cv=cv,  # CV bölmesi.
            scoring="roc_auc",  # Optimizasyon metriği.
            n_jobs=-1  # Tüm çekirdekleri kullan.
        )
        grid.fit(X_train, y_train)  # GridSearch’i eğit.
        best_C = grid.best_params_["C"]  # En iyi C’yi al.

    # ------------------------------------------------------------------------
    # 7.3) Final model eğitimi (best_C ile)
    # ------------------------------------------------------------------------
    model = LogisticRegression(  # Final Logistic Regression modeli.
        penalty=penalty,  # L1/L2.
        C=best_C,  # Seçilen C.
        solver=solver,  # Solver.
        max_iter=2000,  # Yakınsama için.
        random_state=42  # Tekrarlanabilirlik.
    )
    model.fit(X_train, y_train)  # Modeli train verisiyle eğit.

    # ------------------------------------------------------------------------
    # 7.4) Test tahminleri (olasılık + sınıf)
    # ------------------------------------------------------------------------
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]  # Test için p(y=1) olasılıkları.
    y_pred_default = (y_pred_proba_test >= 0.5).astype(int)  # Default threshold=0.5 ile sınıf tahmini.

    # ------------------------------------------------------------------------
    # 7.5) Default threshold metrikleri
    # ------------------------------------------------------------------------
    cm_default = confusion_matrix(y_test, y_pred_default)  # Default eşik confusion matrix.
    tn, fp, fn, tp = cm_default.ravel()  # TN/FP/FN/TP değerlerini aç.

    sens_default = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall, TPR).
    spec_default = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity (TNR).
    acc_default = accuracy_score(y_test, y_pred_default)  # Accuracy.
    ppv_default = tp / (tp + fp) if (tp + fp) > 0 else 0  # PPV (Precision).
    npv_default = tn / (tn + fn) if (tn + fn) > 0 else 0  # NPV.

    auc_test = roc_auc_score(y_test, y_pred_proba_test)  # Test ROC-AUC.

    # ------------------------------------------------------------------------
    # 7.6) Train AUC ve overfitting gap
    # ------------------------------------------------------------------------
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]  # Train olasılık tahmini.
    auc_train = roc_auc_score(y_train, y_pred_proba_train)  # Train ROC-AUC.
    auc_gap = auc_train - auc_test  # Overfitting ölçümü: Train - Test.

    # ------------------------------------------------------------------------
    # 7.7) Cross-validation AUC (final C ile)
    # ------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)  # Aynı CV şemasını kur.
    cv_scores = []  # Her fold AUC’lerini biriktirmek için.

    for tr_idx, val_idx in cv.split(X_train, y_train):  # Train seti içinde CV böl.
        X_tr = X_train.iloc[tr_idx]  # Fold-train X.
        X_val = X_train.iloc[val_idx]  # Fold-val X.
        y_tr = y_train.iloc[tr_idx]  # Fold-train y.
        y_val = y_train.iloc[val_idx]  # Fold-val y.

        cv_model = LogisticRegression(  # Her fold için yeni model.
            penalty=penalty,  # Aynı penalty.
            C=best_C,  # Aynı C.
            solver=solver,  # Aynı solver.
            max_iter=2000,  # Yakınsama için.
            random_state=42  # Tekrarlanabilirlik.
        )
        cv_model.fit(X_tr, y_tr)  # Fold-train ile eğit.
        val_proba = cv_model.predict_proba(X_val)[:, 1]  # Fold-val olasılık tahmini.
        cv_scores.append(roc_auc_score(y_val, val_proba))  # Fold AUC’yi listeye ekle.

    cv_auc_mean = float(np.mean(cv_scores))  # CV AUC ortalaması.
    cv_auc_std = float(np.std(cv_scores))  # CV AUC standart sapması.

    # ------------------------------------------------------------------------
    # 7.8) Youden’s J ile optimal threshold (train üzerinde)
    # ------------------------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba_train)  # Train ROC eğrisi (eşiklerle).

    youden_j = tpr - fpr  # Youden J = Sensitivity - FPR (tpr - fpr).
    best_idx = int(np.argmax(youden_j))  # En iyi J’nin index’i.
    optimal_threshold = float(thresholds[best_idx])  # Optimal threshold değeri.

    y_pred_opt = (y_pred_proba_test >= optimal_threshold).astype(int)  # Testte optimal threshold ile sınıflandır.
    cm_opt = confusion_matrix(y_test, y_pred_opt)  # Optimal threshold confusion matrix.
    tn2, fp2, fn2, tp2 = cm_opt.ravel()  # TN/FP/FN/TP.

    sens_opt = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0  # Optimal threshold sensitivity.
    spec_opt = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0  # Optimal threshold specificity.
    acc_opt = accuracy_score(y_test, y_pred_opt)  # Optimal threshold accuracy.
    ppv_opt = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0  # Optimal threshold PPV.
    npv_opt = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0  # Optimal threshold NPV.

    y_train_pred_opt = (y_pred_proba_train >= optimal_threshold).astype(int)  # Train’de de optimal threshold uygula.
    cm_train_opt = confusion_matrix(y_train, y_train_pred_opt)  # Train confusion matrix (opt threshold).
    tn3, fp3, fn3, tp3 = cm_train_opt.ravel()  # TN/FP/FN/TP.

    sens_train_opt = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else 0  # Train sensitivity (opt).
    spec_train_opt = tn3 / (tn3 + fp3) if (tn3 + fp3) > 0 else 0  # Train specificity (opt).

    # ------------------------------------------------------------------------
    # 7.9) Katsayılar (Beta) ve intercept
    # ------------------------------------------------------------------------
    coefs = model.coef_[0]  # Katsayı vektörü (özellik sırası X_train.columns ile aynı).
    intercept = float(model.intercept_[0])  # Intercept (Beta0).
    feature_names = X_train.columns.tolist()  # Feature isimleri listesi.

    # ------------------------------------------------------------------------
    # 7.10) Sonuç sözlüğü (JSON’a yazmak için)
    # ------------------------------------------------------------------------
    results = {  # Tek modelin bütün sonuçlarını tutacak dict.
        "model_name": model_name,  # Model ismi (Model 1/2).
        "regularization": regularization,  # l1/l2.
        "C": float(best_C),  # Seçilen C.

        "cv_auc_mean": cv_auc_mean,  # CV ortalama AUC.
        "cv_auc_std": cv_auc_std,  # CV std AUC.

        "train_auc": float(auc_train),  # Train AUC.
        "test_auc": float(auc_test),  # Test AUC.
        "auc_gap": float(auc_gap),  # Overfitting gap.

        # Default threshold (0.5) metrikleri:
        "test_accuracy_default": float(acc_default),
        "sensitivity_default": float(sens_default),
        "specificity_default": float(spec_default),
        "ppv_default": float(ppv_default),
        "npv_default": float(npv_default),
        "confusion_matrix_default": cm_default.tolist(),

        # Optimal threshold (Youden) metrikleri:
        "optimal_threshold": optimal_threshold,
        "test_accuracy": float(acc_opt),
        "sensitivity": float(sens_opt),
        "specificity": float(spec_opt),
        "ppv": float(ppv_opt),
        "npv": float(npv_opt),
        "confusion_matrix": cm_opt.tolist(),

        # Train’de optimal threshold metrikleri (kıyas için):
        "sensitivity_train_opt": float(sens_train_opt),
        "specificity_train_opt": float(spec_train_opt),

        # Model parametreleri:
        "intercept": float(intercept),
        "coefficients": {n: float(c) for n, c in zip(feature_names, coefs)},  # Beta katsayıları (feature:beta).

        # ROC çizimi vb. için saklanabilecek ham çıktılar:
        "y_test": y_test.tolist(),
        "y_pred": y_pred_opt.tolist(),
        "y_pred_proba": y_pred_proba_test.tolist(),
    }

    return results, model  # Sonuç sözlüğünü ve modeli geri döndür.

# ============================================================================
# 8) MODELLERİ ÇALIŞTIR (Model1/Model2 x L1/L2)
# ============================================================================

all_results = []  # Tüm modellerin sonuçlarını burada toplayacağız.

print("Model 1 - L1 Regularization (Lasso):")  # Model1-L1 başlığı.
r1_l1, m1_l1 = evaluate_model(X1_train, X1_test, y1_train, y1_test, "Model 1", "l1")  # Eğit+değerlendir.
all_results.append(r1_l1)  # Sonuçları listeye ekle.
print(f"  Optimal C: {r1_l1['C']:.3f}")  # Seçilen C’yi yaz.
print(f"  Optimal Eşik (Youden): {r1_l1['optimal_threshold']:.3f}")  # Optimal threshold.
print(f"  CV AUC: {r1_l1['cv_auc_mean']:.3f} ± {r1_l1['cv_auc_std']:.3f}")  # CV AUC.
print(f"  Train AUC: {r1_l1['train_auc']:.3f}")  # Train AUC.
print(f"  Test AUC: {r1_l1['test_auc']:.3f}")  # Test AUC.
print(f"  AUC Gap: {r1_l1['auc_gap']:.3f}")  # Overfitting gap.
print(f"  Test Acc (Opt): {r1_l1['test_accuracy']:.3f}")  # Optimal eşik accuracy.
print(f"  Sens (Opt): {r1_l1['sensitivity']:.3f}")  # Optimal eşik sensitivity.
print(f"  Spec (Opt): {r1_l1['specificity']:.3f}\n")  # Optimal eşik specificity.

print("Model 1 - L2 Regularization (Ridge):")  # Model1-L2 başlığı.
r1_l2, m1_l2 = evaluate_model(X1_train, X1_test, y1_train, y1_test, "Model 1", "l2")  # Eğit+değerlendir.
all_results.append(r1_l2)  # Sonuçları ekle.
print(f"  Optimal C: {r1_l2['C']:.3f}")  # C.
print(f"  Optimal Eşik (Youden): {r1_l2['optimal_threshold']:.3f}")  # Threshold.
print(f"  CV AUC: {r1_l2['cv_auc_mean']:.3f} ± {r1_l2['cv_auc_std']:.3f}")  # CV AUC.
print(f"  Train AUC: {r1_l2['train_auc']:.3f}")  # Train AUC.
print(f"  Test AUC: {r1_l2['test_auc']:.3f}")  # Test AUC.
print(f"  AUC Gap: {r1_l2['auc_gap']:.3f}")  # Gap.
print(f"  Test Acc (Opt): {r1_l2['test_accuracy']:.3f}")  # Acc.
print(f"  Sens (Opt): {r1_l2['sensitivity']:.3f}")  # Sens.
print(f"  Spec (Opt): {r1_l2['specificity']:.3f}\n")  # Spec.

print("Model 2 - L1 Regularization (Lasso):")  # Model2-L1 başlığı.
r2_l1, m2_l1 = evaluate_model(X2_train, X2_test, y2_train, y2_test, "Model 2", "l1")  # Eğit+değerlendir.
all_results.append(r2_l1)  # Sonuçları ekle.
print(f"  Optimal C: {r2_l1['C']:.3f}")  # C.
print(f"  Optimal Eşik (Youden): {r2_l1['optimal_threshold']:.3f}")  # Threshold.
print(f"  CV AUC: {r2_l1['cv_auc_mean']:.3f} ± {r2_l1['cv_auc_std']:.3f}")  # CV AUC.
print(f"  Train AUC: {r2_l1['train_auc']:.3f}")  # Train AUC.
print(f"  Test AUC: {r2_l1['test_auc']:.3f}")  # Test AUC.
print(f"  AUC Gap: {r2_l1['auc_gap']:.3f}")  # Gap.
print(f"  Test Acc (Opt): {r2_l1['test_accuracy']:.3f}")  # Acc.
print(f"  Sens (Opt): {r2_l1['sensitivity']:.3f}")  # Sens.
print(f"  Spec (Opt): {r2_l1['specificity']:.3f}\n")  # Spec.

print("Model 2 - L2 Regularization (Ridge):")  # Model2-L2 başlığı.
r2_l2, m2_l2 = evaluate_model(X2_train, X2_test, y2_train, y2_test, "Model 2", "l2")  # Eğit+değerlendir.
all_results.append(r2_l2)  # Sonuçları ekle.
print(f"  Optimal C: {r2_l2['C']:.3f}")  # C.
print(f"  Optimal Eşik (Youden): {r2_l2['optimal_threshold']:.3f}")  # Threshold.
print(f"  CV AUC: {r2_l2['cv_auc_mean']:.3f} ± {r2_l2['cv_auc_std']:.3f}")  # CV AUC.
print(f"  Train AUC: {r2_l2['train_auc']:.3f}")  # Train AUC.
print(f"  Test AUC: {r2_l2['test_auc']:.3f}")  # Test AUC.
print(f"  AUC Gap: {r2_l2['auc_gap']:.3f}")  # Gap.
print(f"  Test Acc (Opt): {r2_l2['test_accuracy']:.3f}")  # Acc.
print(f"  Sens (Opt): {r2_l2['sensitivity']:.3f}")  # Sens.
print(f"  Spec (Opt): {r2_l2['specificity']:.3f}\n")  # Spec.

# ============================================================================
# 9) MODEL KARŞILAŞTIRMASI ve EN İYİ MODEL SEÇİMİ
# ============================================================================

print("6. MODEL KARŞILAŞTIRMASI VE EN İYİ MODEL SEÇİMİ")  # Bölüm başlığı.
print("-" * 80)  # Bölüm alt çizgisi.

results_df = pd.DataFrame([  # Sonuçları tablo haline getir.
    {  # Her model için bir satır oluştur.
        "Model": r["model_name"],  # Model 1/2.
        "Regularization": r["regularization"],  # l1/l2.
        "C": r["C"],  # Seçilen C.
        "Optimal Eşik": f"{r['optimal_threshold']:.3f}",  # Optimal threshold formatlı.
        "CV AUC (Mean±Std)": f"{r['cv_auc_mean']:.3f}±{r['cv_auc_std']:.3f}",  # CV AUC metni.
        "Train AUC": f"{r['train_auc']:.3f}",  # Train AUC metni.
        "Test AUC": f"{r['test_auc']:.3f}",  # Test AUC metni.
        "AUC Gap": f"{r['auc_gap']:.3f}",  # Gap metni.
        "Test Accuracy": f"{r['test_accuracy']:.3f}",  # Accuracy metni.
        "Sensitivity": f"{r['sensitivity']:.3f}",  # Sens metni.
        "Specificity": f"{r['specificity']:.3f}",  # Spec metni.
        "PPV": f"{r['ppv']:.3f}",  # PPV metni.
        "NPV": f"{r['npv']:.3f}",  # NPV metni.
    }
    for r in all_results  # all_results listesindeki her model sonucu için.
])

print("\nTÜM MODELLERİN PERFORMANS KARŞILAŞTIRMASI:")  # Başlık.
print(results_df.to_string(index=False))  # DataFrame’i index olmadan yazdır.

best_idx = None  # En iyi modelin index’i (başlangıçta yok).
best_auc = -1  # En iyi test AUC’yi takip etmek için başlangıç.
best_gap = 999  # En iyi gap’i takip etmek için başlangıç.

for i, r in enumerate(all_results):  # Tüm sonuçları dolaş.
    if r["sensitivity"] == 0.0:  # Hiç pCR tahmin etmeyen modeli istemiyoruz.
        print(f"  {r['model_name']} {r['regularization']}: Sensitivity=0 → ATLANDI")  # Bilgilendir.
        continue  # Bu modeli atla.
    if r["auc_gap"] > 0.15:  # Overfitting çok yüksekse (örn >0.15) ele.
        continue  # Atla.

    if (r["test_auc"] > best_auc) or (r["test_auc"] == best_auc and r["auc_gap"] < best_gap):  # Daha iyi kriter mi?
        best_auc = r["test_auc"]  # Best AUC güncelle.
        best_gap = r["auc_gap"]  # Best gap güncelle.
        best_idx = i  # Best index güncelle.

if best_idx is None:  # Eğer filtrelerden geçen model yoksa
    print("\n  Filtrelerden geçen model yok (Sensitivity=0 veya overfitting yüksek).")  # Uyar.
    candidates = [(i, r) for i, r in enumerate(all_results) if r["sensitivity"] > 0]  # Sens>0 olanları topla.
    if len(candidates) > 0:  # Aday varsa
        best_idx = max(candidates, key=lambda t: t[1]["test_auc"])[0]  # En yüksek test AUC olanı seç.
    else:
        best_idx = 0  # Hiçbiri sens>0 değilse ilk modeli fallback olarak seç.

best_model = all_results[best_idx]  # En iyi model sonucunu al.

print("\n EN İYİ MODEL (Optimal Eşik ile):")  # Başlık.
print(f"  Model: {best_model['model_name']}")  # Model adı.
print(f"  Regularization: {best_model['regularization']}")  # Regularization türü.
print(f"  Optimal C: {best_model['C']:.3f}")  # C.
print(f"  Optimal Eşik (Youden): {best_model['optimal_threshold']:.3f}")  # Optimal threshold.
print(f"  Test AUC: {best_model['test_auc']:.3f}")  # Test AUC.
print(f"  AUC Gap: {best_model['auc_gap']:.3f}")  # Gap.
print(f"  Test Accuracy: {best_model['test_accuracy']:.3f}")  # Accuracy.
print(f"  Sensitivity: {best_model['sensitivity']:.3f}")  # Sens.
print(f"  Specificity: {best_model['specificity']:.3f}")  # Spec.
print(f"  PPV: {best_model['ppv']:.3f}")  # PPV.
print(f"  NPV: {best_model['npv']:.3f}")  # NPV.

print("\n   Eşik Karşılaştırması (Default 0.5 vs Optimal):")  # Alt başlık.
print(f"     Default (0.5): Sens={best_model['sensitivity_default']:.3f}, Spec={best_model['specificity_default']:.3f}")  # Default eşik.
print(f"     Optimal ({best_model['optimal_threshold']:.3f}): Sens={best_model['sensitivity']:.3f}, Spec={best_model['specificity']:.3f}")  # Optimal eşik.

# ============================================================================
# 10) SONUÇLARI KAYDETME
# ============================================================================

print("\n7. SONUÇLARI KAYDETME")  # Bölüm başlığı.
print("-" * 80)  # Bölüm alt çizgisi.

with open("model_results.json", "w", encoding="utf-8") as f:  # JSON dosyasını yazma modunda aç.
    json.dump(all_results, f, ensure_ascii=False, indent=2)  # all_results listesini JSON’a yaz.
print("Model sonuçları kaydedildi: model_results.json")  # Bilgilendir.

results_df.to_excel("model_comparison.xlsx", index=False)  # Karşılaştırma tablosunu Excel'e kaydet.
print("Model karşılaştırması kaydedildi: model_comparison.xlsx")  # Bilgilendir.

best_coefs_df = pd.DataFrame({  # En iyi modelin katsayılarını tabloya çevir.
    "Değişken": list(best_model["coefficients"].keys()),  # Feature isimleri.
    "Beta_Katsayı": list(best_model["coefficients"].values()),  # Beta değerleri.
})
best_coefs_df["OR"] = np.exp(best_coefs_df["Beta_Katsayı"])  # OR = exp(beta) hesapla.

best_coefs_df = best_coefs_df.sort_values(  # Mutlak etkiye göre sırala.
    by="Beta_Katsayı",  # Sıralama kriteri beta.
    key=lambda s: s.abs(),  # Mutlak değerle sırala.
    ascending=False  # Büyükten küçüğe.
)

best_coefs_df.to_excel("best_model_coefficients.xlsx", index=False)  # Katsayı tablosunu Excel'e yaz.
print("En iyi model katsayıları kaydedildi: best_model_coefficients.xlsx")  # Bilgilendir.

# ============================================================================
# 11) ÇIKTILARI COLAB’DA İNDİRME
# ============================================================================

files.download("model_results.json")  # JSON çıktısını indir.
files.download("model_comparison.xlsx")  # Model karşılaştırma Excel’ini indir.
files.download("best_model_coefficients.xlsx")  # En iyi model katsayı Excel’ini indir.

print(f"\n{'='*80}")  # Son ayraç çizgisi.
print(" ADIM 3 TAMAMLANDI!")  # Bitiş mesajı.
print(f"{'='*80}")  # Son ayraç çizgisi.


