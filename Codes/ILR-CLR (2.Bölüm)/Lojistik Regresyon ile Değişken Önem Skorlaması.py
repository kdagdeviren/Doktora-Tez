# =====================================================
# Kod: Lojistik Regresyon ile Değişken Önem Skorlaması Aracı
# Amaç:
# - Excel veri setini yüklemek
# - Kullanıcının seçtiği bağımsız değişken (kategorik) ile RCB_ML (0/1) ilişkisini lojistik regresyonla incelemek
# - OR + %95 GA hesaplamak
# - Tedavi yanıtı çapraz tablosu üretmek
# - İstatistiksel anlamlılık + etki büyüklüğü + model açıklayıcılığı + klinik önem ile "önem skoru" hesaplamak
# =====================================================

# =====================================================
# 1) Gerekli kütüphaneler
# =====================================================

import pandas as pd  # Veri okuma/işleme (DataFrame) için.
import numpy as np  # Sayısal işlemler ve OR hesapları için.
import statsmodels.api as sm  # Lojistik regresyon (Logit) ve model özetleri için.

# =====================================================
# 2) Dosya yükleme ve okuma
# =====================================================

try:
    
    from google.colab import files  # Colab dosya yükleme arayüzü.
    print("Lütfen analiz etmek istediğiniz Excel dosyanızı yükleyin:")
    uploaded = files.upload()  # Dosya yükleme penceresini açar.
    file_name = list(uploaded.keys())[0]  # Yüklenen ilk dosyanın adını alır.
    df = pd.read_excel(file_name)  # Excel'i DataFrame olarak okur.
    print(f"'{file_name}' başarıyla yüklendi ve okundu.")
except Exception:
    

# =====================================================
# 3) Veri setinin genel görünümü
# =====================================================

print("\nVeri setinin ilk 5 satırı:")  # Başlık.
print(df.head())  # İlk 5 satırı yazdırır.
print("\nKullanılabilir sütunlar:")  # Başlık.
print(df.columns.tolist())  # Sütun isimlerini listeler.

# =====================================================
# 4) Kullanıcıdan analiz edilecek bağımsız değişkeni alma
# =====================================================

# Kullanıcıdan analiz edeceği sütunu ister.
target_variable = input(
    "\nAnaliz edilecek bağımsız değişkenin adını giriniz (örn: Tubul_i): "
).strip()

# Kullanıcı yanlış sütun adı girerse tekrar ister.
while target_variable not in df.columns:
    print("Bu isimde bir sütun veri setinde bulunamadı. Lütfen doğru sütun adını giriniz.")
    target_variable = input("Bağımsız değişkenin adını tekrar giriniz: ").strip()

# =====================================================
# 5) Kullanıcıdan klinik önem derecesi alma (0/1/2)
# =====================================================

while True:
    try:
        clinical_importance = int(
            input("Bu değişkenin klinik önem derecesini giriniz (0: düşük, 1: orta, 2: yüksek): ")
        )
        if clinical_importance in [0, 1, 2]:
            break
        print("Lütfen 0, 1 veya 2 giriniz.")
    except Exception:
        print("Lütfen sayısal değer giriniz (0/1/2).")

print("\n--- Veri Hazırlığı Başlatılıyor ---")

# =====================================================
# 6) Bağımlı değişken kontrolü: RCB_ML
# =====================================================

# Lojistik regresyonun hedefi RCB_ML (0/1) olmalı.
if "RCB_ML" not in df.columns:
    raise ValueError("HATA: Veri setinde 'RCB_ML' adlı bağımlı değişken yok!")

# Hedef değişkeni alır.
y = df["RCB_ML"]

print("\nRCB_ML dağılımı:")  # Başlık.
print(y.value_counts(dropna=False))  # Sınıf dağılımını gösterir.

# Lojistik regresyon için en az iki sınıf gereklidir.
if y.dropna().nunique() < 2:
    raise ValueError(
        "HATA: RCB_ML değişkeninde yalnızca tek sınıf var. Lojistik regresyon için 0 ve 1 bulunmalı."
    )

# =====================================================
# 7) Bağımsız değişkeni (kategorik) hazırlama
# =====================================================

# Bağımsız değişkeni string'e çevirerek kategorik gibi ele alır.
df[target_variable] = df[target_variable].astype(str)

# Kategorileri kontrol amaçlı yazdırır.
categories = df[target_variable].unique()
print(f"\n{target_variable} sütununda benzersiz kategoriler:")
print(categories)

# Referans kategori: en sık görülen (mode)
reference_category = df[target_variable].mode(dropna=True)[0]
print(f"\nReferans kategori olarak '{reference_category}' seçildi.")

# =====================================================
# 8) Dummy değişkenler (one-hot) oluşturma
# =====================================================

# drop_first=False: tüm dummy'leri oluştur, sonra referans sütunu manuel çıkaracağız.
X = pd.get_dummies(df[[target_variable]], drop_first=False)

# Referans dummy sütunu adı: örn "Tubul_i_2"
ref_col = f"{target_variable}_{reference_category}"

# Referansı çıkararak çoklu doğrusal bağlantıyı (dummy trap) önleriz.
if ref_col in X.columns:
    X = X.drop(columns=[ref_col])
    print(f"Referans dummy sütunu '{ref_col}' silindi.")
else:
    print(f"Uyarı: Referans dummy sütunu '{ref_col}' bulunamadı. (Kategorilerde özel karakter olabilir)")

# Modele intercept eklemek için sabit terim eklenir.
X = sm.add_constant(X)

# statsmodels Logit için float'a çevirmek güvenlidir.
X = X.astype(float)

# =====================================================
# 9) Lojistik regresyon modelini kurma ve çalıştırma
# =====================================================

model = sm.Logit(y, X)  # Lojistik regresyon modeli tanımı.

try:
    result = model.fit(disp=False)  # Modeli eğitir (disp=False: uzun iterasyon çıktısını kapatır).

    print("\nLojistik regresyon sonuç özeti:")
    print(result.summary())  # Model özet tablosu.

    # =====================================================
    # 10) OR ve %95 Güven Aralığı hesaplama
    # =====================================================

    params = result.params  # Katsayılar (log-odds).
    conf = result.conf_int()  # Katsayılar için güven aralığı (log-odds ölçeğinde).

    # OR = exp(beta)
    or_vals = np.exp(params)

    # Güven aralıklarını da OR ölçeğine çevir (exp)
    conf_or = np.exp(conf)
    conf_or.columns = ["Alt CI", "Üst CI"]

    # OR tablosunu birleştir
    or_table = pd.DataFrame({
        "OR": or_vals,
        "Alt CI": conf_or["Alt CI"],
        "Üst CI": conf_or["Üst CI"]
    })

    print("\nOR (Odds Ratio) ve %95 Güven Aralığı:")
    print(or_table)

    # =====================================================
    # 11) Model istatistikleri: Pseudo R² ve LLR p-değeri
    # =====================================================

    pseudo_r2 = result.prsquared  # McFadden Pseudo R²
    llr_p = result.llr_pvalue  # Likelihood ratio test p-değeri

    print(f"\nModel açıklayıcılığı (Pseudo R²): {pseudo_r2:.3f}")
    print(f"Model genel anlamlılığı (LLR p-değeri): {llr_p:.6e}")

    # =====================================================
    # 12) Tedavi yanıtı dağılımı tablosu (cross-tab)
    # =====================================================

    # Kategoriler x RCB_ML (0/1) sayıları
    response_table = pd.crosstab(df[target_variable], df["RCB_ML"], margins=True)
    print("\nTedavi yanıtı dağılımı tablosu:")
    print(response_table)

    # Kolon isimlerini güvenli hale getir (0/1 + All)
    # Not: crosstab margins=True -> "All" kolonu gelir.
    response_table = response_table.rename(columns={0: "RCB_ML_0", 1: "RCB_ML_1", "All": "Toplam"})
    response_table = response_table.rename(index={"All": "Toplam"})

    # Yüzdeler: her kategori içinde (satır bazında) oran
    if "RCB_ML_0" in response_table.columns and "RCB_ML_1" in response_table.columns:
        # Toplam satırını oran hesabına sokmamak için mask kullanılır.
        mask_rows = response_table.index != "Toplam"

        response_table.loc[mask_rows, "% (Tam Yanıt)"] = (
            response_table.loc[mask_rows, "RCB_ML_1"] / response_table.loc[mask_rows, "Toplam"] * 100
        ).round(1)

        response_table.loc[mask_rows, "% (Kısmi Yanıt)"] = (
            response_table.loc[mask_rows, "RCB_ML_0"] / response_table.loc[mask_rows, "Toplam"] * 100
        ).round(1)

    print("\nYanıt yüzdeleri eklenmiş tablo:")
    print(response_table)

    # =====================================================
    # 13) Değişken önem skoru hesaplama
    # =====================================================

    print("\n--- Değişken Önem Skalası Hesaplanıyor ---")

    # En küçük p-değeri (değişkenin en güçlü kategorisi üzerinden)
    min_p = result.pvalues.min()

    # En büyük OR etkisi:
    # - OR'lar 1'in üstünde veya altında olabilir
    # - simetrik etki için max(OR, 1/OR) alınır
    or_non_const = or_vals.drop(labels=["const"], errors="ignore")
    if len(or_non_const) > 0:
        max_or_effect = float(max(or_non_const.max(), 1 / or_non_const.min()))
    else:
        max_or_effect = 0.0

    # --- İstatistiksel anlamlılık puanı (0-3)
    if min_p < 0.001:
        stat_score = 3
    elif min_p < 0.01:
        stat_score = 2
    elif min_p < 0.05:
        stat_score = 1
    else:
        stat_score = 0

    # --- Etki büyüklüğü puanı (0-3)
    if max_or_effect > 5:
        effect_score = 3
    elif max_or_effect >= 2:
        effect_score = 2
    elif max_or_effect >= 1.2:
        effect_score = 1
    else:
        effect_score = 0

    # --- Model açıklayıcılığı puanı (0-2)
    if pseudo_r2 >= 0.2 and llr_p < 0.01:
        model_score = 2
    elif (pseudo_r2 >= 0.05) or (llr_p < 0.05):
        model_score = 1
    else:
        model_score = 0

    # --- Klinik önem puanı (0-2) kullanıcı girdisi
    clinical_score = clinical_importance

    # Toplam skor (0-10)
    total_score = stat_score + effect_score + model_score + clinical_score

    # Skora göre sınıflama
    if total_score >= 9:
        category = "Mükemmel"
    elif total_score >= 7:
        category = "İyi"
    elif total_score >= 5:
        category = "Orta"
    elif total_score >= 3:
        category = "Zayıf"
    else:
        category = "Etkisiz"

    print("\nSkala Kriterleri ve Puanları:")
    print(f"  - İstatistiksel Anlamlılık (min p={min_p:.4g})      : {stat_score}/3")
    print(f"  - İlişki Gücü (en etkili OR etkisi={max_or_effect:.3f}): {effect_score}/3")
    print(f"  - Model Açıklayıcılığı (Pseudo R²={pseudo_r2:.3f}, LLR p={llr_p:.3g}): {model_score}/2")
    print(f"  - Klinik Önem (Kullanıcı Girdisi)                   : {clinical_score}/2")
    print("-" * 55)
    print(f"Toplam Ham Puan: {total_score}/10")
    print(f"NİHAİ ÖNEM SKORU: {total_score:.1f}/10  →  {category} öngörücü")
    print("-" * 55)

except Exception as e:
    print("\nModel kurulurken hata oluştu:")
    print(str(e))

# =====================================================
# Kod Sonu - Lojistik Regresyon ile Değişken Önem Skoru Hesaplandı
# =====================================================


