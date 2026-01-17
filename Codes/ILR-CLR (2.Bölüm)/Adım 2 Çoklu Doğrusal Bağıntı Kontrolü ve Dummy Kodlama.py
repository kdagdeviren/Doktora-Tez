# =============================================================================
# ÇOKLU LOJİSTİK REGRESYON ANALİZİ - ADIM 2: MULTICOLLINEARITY KONTROLÜ (VIF)
# =============================================================================
# Amaç:
# 1) Kategorik değişkenleri dummy kodlamak (referans kategori: en sık görülen)
# 2) VIF (Variance Inflation Factor) hesaplamak
# 3) Perfect multicollinearity tespit etmek (VIF >= 999)
# 4) Yüksek VIF’li değişkenler arasında korelasyon çiftlerini raporlamak
# 5) Sonuçları CSV/JSON olarak kaydetmek
# ============================================================================

# ============================================================================
# 0) GEREKLİ KÜTÜPHANELER
# ============================================================================

import json  # JSON dosyalarını okumak/yazmak için.
import numpy as np  # Sayısal işlemler ve NaN/inf kontrolleri için.
import pandas as pd  # Excel okuma ve DataFrame işlemleri için.

from google.colab import files  # SADECE Colab dosya yükleme/indirme arayüzü için.

from statsmodels.tools.tools import add_constant  # VIF hesabında sabit terim eklemek için.
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF hesaplamak için.

#=============================================================================
# 1) PROGRAM BAŞLIĞI
#=============================================================================

print("=" * 80)  # Konsolda okunabilirlik için ayraç.
print("ÇOKLU LOJİSTİK REGRESYON - ADIM 2: MULTICOLLINEARITY KONTROLÜ (VIF)")
print("=" * 80)
print()  # Boş satır.

# =============================================================================
# 2) VERİ YÜKLEME (DİREKT COLAB)
# =============================================================================
# Bu kod sadece Colab ortamına göre düzenlenmiştir.

print("1. VERİ YÜKLEME")  # Bölüm başlığı.
print("-" * 80)

# (Opsiyonel) ADIM 1 çıktısı (coklu_lr_hazirlik.json) varsa okur.
# Bu dosya bulunmazsa kod durmaz; sadece bilgilendirir.
prep_data = None  # Varsayılan: yok.

try:
    with open("coklu_lr_hazirlik.json", "r", encoding="utf-8") as f:
        prep_data = json.load(f)  # Hazırlık verilerini yükler.
    print("Hazırlık verileri yüklendi: coklu_lr_hazirlik.json")
except FileNotFoundError:
    print("Hazırlık dosyası bulunamadı (coklu_lr_hazirlik.json) → devam ediliyor.")

print()  # Boş satır.

# Excel dosyasını yüklet.
print("Lütfen Excel dosyanızı yükleyin:")  # Kullanıcı talimatı.
uploaded = files.upload()  # Colab dosya yükleme penceresi.
excel_path = list(uploaded.keys())[0]  # Yüklenen ilk dosyanın adı.

# Excel’i DataFrame olarak oku.
data = pd.read_excel(excel_path)

# Veri boyut bilgisini yazdır.
print(f"Veri yüklendi: {data.shape[0]} satır (hasta), {data.shape[1]} sütun")
print()

# Bağımlı değişken adı (hedef).
target = "RCB_ML"

# =============================================================================
# 3) DEĞİŞKEN LİSTELERİ (Model 1 & Model 2)
# =============================================================================
# Model 1 ve Model 2, dummy kodlama ve VIF analizinde kullanılacak bağımsız değişken setlerini temsil eder.

print("2. DEĞİŞKEN LİSTELERİ")
print("-" * 80)

# Model 1: Önem skoru >= 7 olan değişkenler (17 değişken).
model1_vars = {
    "i1": "Histolojik Tip",
    "i2": "ER Durumu",
    "i3": "PR Durumu",
    "i4": "HER2 Durumu",
    "i5": "Moleküler Tip",
    "i6": "Ki-67 İndeksi",
    "i7": "Tübül Derecesi",
    "i8": "Nükleer Derece",
    "i10": "Histolojik Grade",
    "i12": "TIL Seviyesi",
    "i13": "Metastaz Durumu",
    "i14": "Metastaz Yeri",
    "i15": "Tanı Evresi",
    "i53": "Kitle Şekli",               
    "i54": "Kitle Konturu",             
    "i55": "Kitle Dansitesi",           
    "i56": "Kalsifikasyon Morfolojisi", 
}

# Model 2: Model 1 + 2 ek değişken (toplam 19 değişken).
model2_vars = model1_vars.copy()
model2_vars.update({
    "i48": "BI-RADS Sınıflandırması",
    "i19": "Kan Grubu",
})

print(f"Model 1 değişken sayısı: {len(model1_vars)}")
print(f"Model 2 değişken sayısı: {len(model2_vars)}")
print()

# =============================================================================
# 4) REFERANS KATEGORİ SEÇİMİ (EN SIK GÖRÜLEN)
# =============================================================================
# Dummy kodlamada her kategorik değişkenden 1 kategori referans seçilir ve dummy kolonları oluşturulurken bu referans
# dummy sütunu matrise dahil edilmez (dummy trap / perfect multicollinearity riskini azaltmak için).

print("3. REFERANS KATEGORİ SEÇİMİ (EN SIK GÖRÜLEN)")
print("-" * 80)

reference_categories = {}  # Her değişken için seçilen referans kategoriyi tutar.

# Model 1 + Model 2 ekleri (i48, i19) birlikte dolaşılır.
for var_code in list(model1_vars.keys()) + ["i48", "i19"]:
    if var_code not in data.columns:
        # Veri setinde yoksa pas geçilir.
        continue

    # NaN hariç frekansları hesapla.
    vc = data[var_code].value_counts(dropna=True)

    if len(vc) == 0:
        continue  # Tamamen boşsa pas.

    # En sık görülen kategori.
    ref_cat = vc.idxmax()
    ref_count = vc.max()

    # Bazı kategoriler string olabilir; int dönüşümü mümkün değilse orijinal haliyle saklanır.
    try:
        reference_categories[var_code] = int(ref_cat)
    except Exception:
        reference_categories[var_code] = ref_cat

    var_name = model1_vars.get(var_code, model2_vars.get(var_code, var_code))
    print(f"  {var_code} ({var_name}): Referans = {ref_cat} (n={ref_count})")

print()
print(f" Referans kategori seçilen değişken sayısı: {len(reference_categories)}")
print()

# =============================================================================
# 5) DUMMY KODLAMA FONKSİYONU
# =============================================================================
# Amaç:
# - i1, i2, ... gibi kategorik değişkenleri dummy (one-hot) kolonlara dönüştürmek
# - Seçilen referans kategorinin dummy sütununu çıkarmak
# - Kolon isimlerini daha okunur bir formatta üretmek: i1_cat2 gibi

print("4. DUMMY KODLAMA")
print("-" * 80)

def create_dummy_variables(df_in, var_dict, ref_dict):
    """
    Kategorik değişkenleri dummy (one-hot) kodlar ve referans kategoriyi çıkarır.

    Parametreler:
      df_in      : pd.DataFrame -> Orijinal veri seti
      var_dict   : dict         -> { 'i1': 'Histolojik Tip', ... } gibi
      ref_dict   : dict         -> { 'i1': 2, 'i2': 1, ... } gibi

    Dönenler:
      X_dummy    : pd.DataFrame -> Dummy kodlanmış tasarım matrisi
      dummy_info : dict         -> Her değişkenin dummy kolon bilgileri (JSON'a kaydetmek için)
    """
    # Aynı satır index’i ile başlatmak concat sırasında hizalama sorununu engeller.
    X_dummy = pd.DataFrame(index=df_in.index)

    dummy_info = {}  # Her var_code için metadata tutulur.

    for var_code, var_name in var_dict.items():
        # Değişken veri setinde yoksa geç.
        if var_code not in df_in.columns:
            continue

        # NaN hariç kategorileri al.
        categories = sorted(df_in[var_code].dropna().unique())

        # Hiç kategori yoksa geç.
        if len(categories) == 0:
            continue

        # Referans kategori: ref_dict'te varsa onu kullan, yoksa ilk kategoriyi kullan.
        ref_cat = ref_dict.get(var_code, categories[0])

        # Dummy kodlama:
        # drop_first=False -> tüm kategorileri üret
        dummy_cols = pd.get_dummies(df_in[var_code], prefix=var_code, drop_first=False).astype(int)

        # Referans dummy sütununun ham adı: örn i1_2
        ref_col_name = f"{var_code}_{ref_cat}"

        # Referans sütun varsa çıkar.
        if ref_col_name in dummy_cols.columns:
            dummy_cols = dummy_cols.drop(columns=[ref_col_name])

        # Sütun adlarını daha okunur yap:
        # i1_3 -> i1_cat3
        rename_map = {}
        for col in dummy_cols.columns:
            parts = col.split("_", 1)
            cat_part = parts[1] if len(parts) > 1 else "NA"
            rename_map[col] = f"{var_code}_cat{cat_part}"
        dummy_cols = dummy_cols.rename(columns=rename_map)

        # Güvenlik:
        # Teorik olarak bazı kategoriler görünmüyor olabilir; eksik dummy sütunlarını sıfır ile ekle.
        
        for cat in categories:
            if cat == ref_cat:
                continue
            col_name = f"{var_code}_cat{cat}"
            if col_name not in dummy_cols.columns:
                dummy_cols[col_name] = 0

        # Kolonları alfabetik sırala (çıktının düzenli olması için).
        dummy_cols = dummy_cols[sorted(dummy_cols.columns)]

        # Büyük tasarım matrisi ile birleştir.
        X_dummy = pd.concat([X_dummy, dummy_cols], axis=1)

        # JSON raporuna yazılacak bilgi.
        dummy_info[var_code] = {
            "name": var_name,
            "reference": ref_cat,
            "categories": [c for c in categories],
            "dummy_columns": dummy_cols.columns.tolist(),
        }

    return X_dummy, dummy_info

# Model 1 dummy matrisi.
print("Model 1 için dummy kodlama yapılıyor...")
X1_dummy, dummy_info1 = create_dummy_variables(data, model1_vars, reference_categories)
print(f"Model 1 dummy sütun sayısı: {X1_dummy.shape[1]}")

# Model 2 dummy matrisi.
print("\nModel 2 için dummy kodlama yapılıyor...")
X2_dummy, dummy_info2 = create_dummy_variables(data, model2_vars, reference_categories)
print(f"Model 2 dummy sütun sayısı: {X2_dummy.shape[1]}")
print()

# =============================================================================
# 6) VIF HESAPLAMA FONKSİYONU
# =============================================================================
# VIF yorumlama:
# - VIF ~ 1: multicollinearity yok/çok düşük
# - VIF 5+ : dikkat (korelasyon olabilir)
# - VIF 10+ : yüksek multicollinearity
# - VIF 999 (bu kodda): inf/nan/perfect collinearity olarak işaretlenen durumlar

print("5. VIF HESAPLAMA")
print("-" * 80)

def calculate_vif(X):
    """
    Variance Inflation Factor (VIF) hesaplar.

    Notlar:
    - add_constant ile const eklenir.
    - VIF yalnızca bağımsız değişkenler için döndürülür (const hariç).
    - VIF inf/nan ya da hesaplanamazsa 999.0 yazılır.
    """
    # statsmodels hesaplamasında float daha stabil.
    X_numeric = X.astype(float)

    # Sabit terim ekle (const).
    X_with_const = add_constant(X_numeric)

    # Çıktı tablosu.
    vif_data = pd.DataFrame({"Değişken": X_numeric.columns})

    vif_values = []

    # variance_inflation_factor:
    # - X_with_const.values içinde 0. kolon const
    # - Bu yüzden i+1 ile her değişkenin VIF'ini alıyoruz
    for i in range(X_numeric.shape[1]):
        try:
            vif_val = variance_inflation_factor(X_with_const.values, i + 1)

            # inf/nan durumlarını "perfect multicollinearity" gibi işaretlemek için 999.0'a çeviriyoruz.
            if (not np.isfinite(vif_val)) or np.isnan(vif_val):
                vif_val = 999.0

            vif_values.append(float(vif_val))
        except Exception:
            # Herhangi bir hesap hatasında 999.0 yaz.
            vif_values.append(999.0)

    vif_data["VIF"] = vif_values
    return vif_data

# Model 1 VIF hesapla.
print("Model 1 için VIF hesaplanıyor...")
vif1 = calculate_vif(X1_dummy)
print("Model 1 VIF hesaplandı")

# Model 2 VIF hesapla.
print("\nModel 2 için VIF hesaplanıyor...")
vif2 = calculate_vif(X2_dummy)
print("Model 2 VIF hesaplandı")
print()

# =============================================================================
# 7) VIF SONUÇLARINI YORUMLAMA
# =============================================================================

print("6. VIF SONUÇLARI VE YORUMLAMA")
print("-" * 80)

def interpret_vif(vif_data, model_name, threshold=5.0):
    """
    VIF sonuçlarını üç grupta raporlar:
    1) Perfect multicollinearity : VIF >= 999
    2) Yüksek VIF                : threshold <= VIF < 999
    3) Normal VIF                : VIF < threshold

    Dönen:
      perfect_vars : list[str]
      high_vars    : list[str]
    """
    print(f"\n{model_name} - VIF Özeti")
    print("-" * 80)

    # Perfect multicollinearity.
    perfect = vif_data[vif_data["VIF"] >= 999.0]
    if len(perfect) > 0:
        print(f"🚨 PERFECT MULTICOLLINEARITY (VIF >= 999): {len(perfect)} değişken")
        for _, row in perfect.iterrows():
            print(f"   - {row['Değişken']} | VIF={row['VIF']:.1f}")

    # High VIF.
    high = vif_data[(vif_data["VIF"] >= threshold) & (vif_data["VIF"] < 999.0)]
    if len(high) > 0:
        print(f"\n  YÜKSEK VIF ({threshold} <= VIF < 999): {len(high)} değişken")
        for _, row in high.sort_values("VIF", ascending=False).iterrows():
            print(f"   - {row['Değişken']} | VIF={row['VIF']:.2f}")

    # Normal VIF.
    normal = vif_data[vif_data["VIF"] < threshold]
    print(f"\n NORMAL VIF (VIF < {threshold}): {len(normal)} değişken")

    return perfect["Değişken"].tolist(), high["Değişken"].tolist()

# Model 1 yorumla.
perfect_multicoll1, high_vif1 = interpret_vif(vif1, "Model 1", threshold=5.0)
print()

# Model 2 yorumla.
perfect_multicoll2, high_vif2 = interpret_vif(vif2, "Model 2", threshold=5.0)
print()

# =============================================================================
# 8) YÜKSEK VIF’Lİ DEĞİŞKENLER İÇİN KORELASYON ANALİZİ
# =============================================================================
# Not:
# - Dummy değişkenler 0/1 olduğu için korelasyon anlamlıdır.
# - |r| > 0.7 gibi bir eşik, yüksek benzerlik/çakışma gösterebilir.

print("7. KORELASYON MATRİSİ ANALİZİ")
print("-" * 80)

def analyze_correlations(X_dummy, high_vif_vars, model_name, correlation_threshold=0.7):
    """
    high_vif_vars içindeki değişkenlerin korelasyon matrisini hesaplar.
    |r| > correlation_threshold olan çiftleri raporlar.

    Dönen:
      high_pairs: list[dict] -> {'Değişken1':..., 'Değişken2':..., 'Korelasyon':...}
    """
    if len(high_vif_vars) == 0:
        print(f"\n{model_name}: Yüksek VIF yok → korelasyon analizi gerekmiyor.")
        return []

    # Gerçekte matriste var olanları filtrele.
    available = [v for v in high_vif_vars if v in X_dummy.columns]

    if len(available) < 2:
        print(f"\n{model_name}: Korelasyon analizi için yeterli değişken yok.")
        return []

    print(f"\n{model_name} - Yüksek Korelasyonlu Çiftler (|r| > {correlation_threshold})")
    print("-" * 80)

    corr_matrix = X_dummy[available].corr()
    high_pairs = []

    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v1, v2 = cols[i], cols[j]
            r = corr_matrix.iloc[i, j]

            if abs(r) > correlation_threshold:
                high_pairs.append({
                    "Değişken1": v1,
                    "Değişken2": v2,
                    "Korelasyon": float(r),
                })
                print(f"   {v1} ↔ {v2} | r={r:.3f}")

    if len(high_pairs) == 0:
        print("   Yüksek korelasyonlu çift bulunamadı.")

    return high_pairs

# Model 1 korelasyon taraması.
high_corr1 = analyze_correlations(X1_dummy, high_vif1, "Model 1", correlation_threshold=0.7)
print()

# Model 2 korelasyon taraması.
high_corr2 = analyze_correlations(X2_dummy, high_vif2, "Model 2", correlation_threshold=0.7)
print()

# =============================================================================
# 9) SONUÇLARI KAYDETME (CSV + JSON)
# =============================================================================
# Çıktılar:
# - model1_vif_results.csv
# - model2_vif_results.csv
# - vif_summary_report.json
# - dummy_coding_info.json

print("=" * 80)
print("8. KAYDETME ve ÖZET RAPOR")
print("=" * 80)

def convert_to_python_types(obj):
    """
    JSON’a yazarken NumPy tiplerini Python tiplerine çevirir.
    Örn:
      np.int64  -> int
      np.float64 -> float
      np.ndarray -> list
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    return obj

# VIF sonuçlarını CSV olarak kaydet.
vif1.to_csv("model1_vif_results.csv", index=False)
vif2.to_csv("model2_vif_results.csv", index=False)

print("\n VIF sonuçları kaydedildi:")
print("   - model1_vif_results.csv")
print("   - model2_vif_results.csv")

# Özet rapor oluştur.
vif_summary = {
    "model1": {
        "total_variables": int(len(vif1)),
        "perfect_multicollinearity": int(len(perfect_multicoll1)),
        "high_vif": int(len(high_vif1)),
        "normal_vif": int((vif1["VIF"] < 5.0).sum()),
        "perfect_multicoll_vars": perfect_multicoll1,
        "high_vif_vars": high_vif1,
        "high_corr_pairs": high_corr1,
    },
    "model2": {
        "total_variables": int(len(vif2)),
        "perfect_multicollinearity": int(len(perfect_multicoll2)),
        "high_vif": int(len(high_vif2)),
        "normal_vif": int((vif2["VIF"] < 5.0).sum()),
        "perfect_multicoll_vars": perfect_multicoll2,
        "high_vif_vars": high_vif2,
        "high_corr_pairs": high_corr2,
    },
    "reference_categories": convert_to_python_types(reference_categories),
}

# JSON olarak yaz.
with open("vif_summary_report.json", "w", encoding="utf-8") as f:
    json.dump(vif_summary, f, ensure_ascii=False, indent=2)

print("VIF özet raporu kaydedildi: vif_summary_report.json")

# Dummy kodlama bilgilerini yaz.
dummy_info_all = {
    "model1": convert_to_python_types(dummy_info1),
    "model2": convert_to_python_types(dummy_info2),
    "reference_categories": convert_to_python_types(reference_categories),
}

with open("dummy_coding_info.json", "w", encoding="utf-8") as f:
    json.dump(dummy_info_all, f, ensure_ascii=False, indent=2)

print("Dummy kodlama bilgileri kaydedildi: dummy_coding_info.json")
print()

# =============================================================================
# 10) ÇIKTILARI COLAB’DA İNDİRME
# =============================================================================
# Eğer çıktıları otomatik indirmek istersen aşağıdaki satırların başındaki # işaretlerini kaldır.

files.download("model1_vif_results.csv")
files.download("model2_vif_results.csv")
files.download("vif_summary_report.json")
files.download("dummy_coding_info.json")

# =============================================================================
# 11) KONSOL ÖZETİ
# =============================================================================

print("\n" + "=" * 80)
print("ÖZET")
print("=" * 80)

print("\nModel 1:")
print(f"  Toplam dummy değişken: {len(vif1)}")
print(f"  Perfect multicollinearity (VIF>=999): {len(perfect_multicoll1)}")
print(f"  Yüksek VIF (5-999): {len(high_vif1)}")
print(f"  Normal VIF (<5): {(vif1['VIF'] < 5.0).sum()}")

print("\nModel 2:")
print(f"  Toplam dummy değişken: {len(vif2)}")
print(f"  Perfect multicollinearity (VIF>=999): {len(perfect_multicoll2)}")
print(f"  Yüksek VIF (5-999): {len(high_vif2)}")
print(f"  Normal VIF (<5): {(vif2['VIF'] < 5.0).sum()}")

print("\n" + "=" * 80)
print(" ADIM 2 TAMAMLANDI!")
print(" Sonraki adım: ADIM 3 - Model Kurulumu ve Değerlendirme")
print("=" * 80)


