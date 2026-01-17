import pandas as pd
import numpy as np
import statsmodels.api as sm

try:
    
    from google.colab import files
    print("Lütfen analiz etmek istediğiniz Excel dosyanızı yükleyin:")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_excel(file_name)
    print(f"'{file_name}' başarıyla yüklendi ve okundu.")
except Exception:
    

print("\nVeri setinin ilk 5 satırı:")
print(df.head())
print("\nKullanılabilir sütunlar:")
print(df.columns.tolist())

target_variable = input(
    "\nAnaliz edilecek bağımsız değişkenin adını giriniz (örn: Tubul_i): "
).strip()

while target_variable not in df.columns:
    print("Bu isimde bir sütun veri setinde bulunamadı. Lütfen doğru sütun adını giriniz.")
    target_variable = input("Bağımsız değişkenin adını tekrar giriniz: ").strip()

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

if "RCB_ML" not in df.columns:
    raise ValueError("HATA: Veri setinde 'RCB_ML' adlı bağımlı değişken yok!")

y = df["RCB_ML"]

print("\nRCB_ML dağılımı:")
print(y.value_counts(dropna=False))

if y.dropna().nunique() < 2:
    raise ValueError(
        "HATA: RCB_ML değişkeninde yalnızca tek sınıf var. Lojistik regresyon için 0 ve 1 bulunmalı."
    )

df[target_variable] = df[target_variable].astype(str)

categories = df[target_variable].unique()
print(f"\n{target_variable} sütununda benzersiz kategoriler:")
print(categories)

reference_category = df[target_variable].mode(dropna=True)[0]
print(f"\nReferans kategori olarak '{reference_category}' seçildi.")

X = pd.get_dummies(df[[target_variable]], drop_first=False)

ref_col = f"{target_variable}_{reference_category}"

if ref_col in X.columns:
    X = X.drop(columns=[ref_col])
    print(f"Referans dummy sütunu '{ref_col}' silindi.")
else:
    print(f"Uyarı: Referans dummy sütunu '{ref_col}' bulunamadı. (Kategorilerde özel karakter olabilir)")

X = sm.add_constant(X)

X = X.astype(float)

model = sm.Logit(y, X)

try:
    result = model.fit(disp=False)

    print("\nLojistik regresyon sonuç özeti:")
    print(result.summary())

    params = result.params
    conf = result.conf_int()

    or_vals = np.exp(params)

    conf_or = np.exp(conf)
    conf_or.columns = ["Alt CI", "Üst CI"]

    or_table = pd.DataFrame({
        "OR": or_vals,
        "Alt CI": conf_or["Alt CI"],
        "Üst CI": conf_or["Üst CI"]
    })

    print("\nOR (Odds Ratio) ve %95 Güven Aralığı:")
    print(or_table)

    pseudo_r2 = result.prsquared
    llr_p = result.llr_pvalue

    print(f"\nModel açıklayıcılığı (Pseudo R²): {pseudo_r2:.3f}")
    print(f"Model genel anlamlılığı (LLR p-değeri): {llr_p:.6e}")

    response_table = pd.crosstab(df[target_variable], df["RCB_ML"], margins=True)
    print("\nTedavi yanıtı dağılımı tablosu:")
    print(response_table)

    response_table = response_table.rename(columns={0: "RCB_ML_0", 1: "RCB_ML_1", "All": "Toplam"})
    response_table = response_table.rename(index={"All": "Toplam"})

    if "RCB_ML_0" in response_table.columns and "RCB_ML_1" in response_table.columns:
        mask_rows = response_table.index != "Toplam"

        response_table.loc[mask_rows, "% (Tam Yanıt)"] = (
            response_table.loc[mask_rows, "RCB_ML_1"] / response_table.loc[mask_rows, "Toplam"] * 100
        ).round(1)

        response_table.loc[mask_rows, "% (Kısmi Yanıt)"] = (
            response_table.loc[mask_rows, "RCB_ML_0"] / response_table.loc[mask_rows, "Toplam"] * 100
        ).round(1)

    print("\nYanıt yüzdeleri eklenmiş tablo:")
    print(response_table)

    print("\n--- Değişken Önem Skalası Hesaplanıyor ---")

    min_p = result.pvalues.min()

    or_non_const = or_vals.drop(labels=["const"], errors="ignore")
    if len(or_non_const) > 0:
        max_or_effect = float(max(or_non_const.max(), 1 / or_non_const.min()))
    else:
        max_or_effect = 0.0

    if min_p < 0.001:
        stat_score = 3
    elif min_p < 0.01:
        stat_score = 2
    elif min_p < 0.05:
        stat_score = 1
    else:
        stat_score = 0

    if max_or_effect > 5:
        effect_score = 3
    elif max_or_effect >= 2:
        effect_score = 2
    elif max_or_effect >= 1.2:
        effect_score = 1
    else:
        effect_score = 0

    if pseudo_r2 >= 0.2 and llr_p < 0.01:
        model_score = 2
    elif (pseudo_r2 >= 0.05) or (llr_p < 0.05):
        model_score = 1
    else:
        model_score = 0

    clinical_score = clinical_importance

    total_score = stat_score + effect_score + model_score + clinical_score

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
