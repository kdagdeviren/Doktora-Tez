import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    f1_score, roc_curve, auc, classification_report,
    precision_recall_fscore_support, brier_score_loss
)
from sklearn.preprocessing import label_binarize, LabelEncoder

from sklearn.inspection import permutation_importance

from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE

from scipy.stats import wilcoxon

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PET VERİLERİ - EKSİK ANALİZLER VE İSTATİSTİKSEL TESTLER")
print("="*80)

from google.colab import files

print("\n Excel dosyanızı yükleyin:")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]

data = pd.read_excel(file_name)

print(f" Veri yüklendi: {data.shape}")

pet_features = [
    'SUVmax', 'SUVmean4', 'TLG', 'MTV',
    'Yüzey/Hacim Oranı4', 'Küresellik4', 'Asferisite4',
    'SUV Varyansı4', 'SUV Eğriliği4',
    'GLCM Entropi4', 'GLCM Kontrast4',
    'GLRLM Non-Uniformite4', 'NGTDM Coarseness4', 'GLSZM Entropi4'
]

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

target = 'RCB_Kategorize'
print(" Veri tipleri kontrol ediliyor ve düzeltiliyor...")
for feat in pet_features + all_features:
    if feat in data.columns:
        data[feat] = pd.to_numeric(data[feat].astype(str).str.replace(',', '.'), errors='coerce')

data = data.dropna(subset=[target])

for feat in pet_features + all_features:
    if feat in data.columns:
        data[feat] = data[feat].fillna(data[feat].mean())
data_pet = data.copy()
print(f"\n PET verisi olan hasta sayısı: {len(data_pet)}")

le = LabelEncoder()

data_pet['RCB_encoded'] = le.fit_transform(data_pet[target])

print(f" RCB sınıfları: {le.classes_}")

X_all = data_pet[all_features]
X_pet = data_pet[pet_features]
X_all_pet = data_pet[all_features + pet_features]

y = data_pet['RCB_encoded']

print("\n" + "="*80)
print(" SINIF DAĞILIMI ANALİZİ")
print("="*80)

X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_all_pet = X_all_pet.loc[X_train_all.index]
X_test_all_pet = X_all_pet.loc[X_test_all.index]

print("\n TÜM VERİ SETİ (PET kohortu):")

class_dist_full = data_pet[target].value_counts().sort_index()

for cls, count in class_dist_full.items():
    pct = count / len(data_pet) * 100
    print(f"  {cls}: {count} hasta ({pct:.1f}%)")

print("\n TRAIN SETİ:")

train_dist = pd.Series(y_train).value_counts().sort_index()

for cls_encoded, count in train_dist.items():
    cls_name = le.inverse_transform([cls_encoded])[0]
    pct = count / len(y_train) * 100
    print(f"  {cls_name}: {count} hasta ({pct:.1f}%)")

print("\n TEST SETİ:")

test_dist = pd.Series(y_test).value_counts().sort_index()

for cls_encoded, count in test_dist.items():
    cls_name = le.inverse_transform([cls_encoded])[0]
    pct = count / len(y_test) * 100
    print(f"  {cls_name}: {count} hasta ({pct:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (data_y, title) in zip(
    axes,
    [
        (y, "Tüm Veri Seti"),
        (y_train, "Train Seti"),
        (y_test, "Test Seti"),
    ]
):
    counts = pd.Series(data_y).value_counts().sort_index()

    labels = [le.inverse_transform([i])[0] for i in counts.index]

    ax.bar(
        labels,
        counts.values,
        color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"][: len(labels)]
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Hasta Sayısı")
    ax.set_xlabel("RCB Sınıfı")

    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontweight="bold")

plt.tight_layout()

plt.savefig("sinif_dagilimi_analizi.png", dpi=300, bbox_inches="tight")
print("\n Grafik kaydedildi: sinif_dagilimi_analizi.png")

plt.show()

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

    outer_cv = StratifiedKFold(
        n_splits=n_outer,
        shuffle=True,
        random_state=random_state
    )

    outer_scores = {
        "test_auc": [],
        "test_accuracy": [],
        "test_f1_macro": []
    }

    print(f"\n🔍 {model_name} için Nested CV başlatılıyor...")
    print(f"Outer folds: {n_outer}, Inner folds: {n_inner}\n")

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train_outer = X.iloc[train_idx]
        X_test_outer  = X.iloc[test_idx]
        y_train_outer = y.iloc[train_idx]
        y_test_outer  = y.iloc[test_idx]

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

            model_inner = clone(model_template)

            model_inner.fit(X_train_inner, y_train_inner)

            y_val_proba = model_inner.predict_proba(X_val_inner)

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
                auc_inner = 0.5

            inner_auc_scores.append(auc_inner)

        inner_auc_mean = np.mean(inner_auc_scores)
        inner_auc_std  = np.std(inner_auc_scores)

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

        gap = inner_auc_mean - auc_outer

        print(
            f"Fold {outer_fold}: "
            f"Inner AUC={inner_auc_mean:.3f}±{inner_auc_std:.3f} | "
            f"Outer AUC={auc_outer:.3f} | "
            f"Gap={gap:.3f}"
        )

    print(f"\n {model_name} — Nested CV Özet:")

    for metric, values in outer_scores.items():
        mean_val = np.mean(values)
        std_val  = np.std(values)
        print(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}")

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

lgb_template = lgb.LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1)
nested_scores_all_pet = nested_cv_evaluation(
    X_all_pet,
    y,
    lgb_template,
    model_name="Model ALL+PET (LightGBM)"
)

print("\n" + "="*80)
print(" BOOTSTRAP AUC + İSTATİSTİKSEL KARŞILAŞTIRMA")
print("="*80)

from scipy.stats import wilcoxon

def bootstrap_auc(
    X_test,
    y_test,
    model_fitted,
    n_iterations=500,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    auc_scores = []

    y_true = y_test.values if hasattr(y_test, "values") else y_test

    classes_sorted = np.unique(y_true)

    for i in range(n_iterations):
        idx = rng.choice(len(X_test), size=len(X_test), replace=True)

        X_boot = X_test.iloc[idx]
        y_boot = y_true[idx]

        y_proba = model_fitted.predict_proba(X_boot)

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
            continue

    return np.array(auc_scores)

model_all_final = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model_all_final.fit(X_train_all, y_train)

model_all_pet_final = lgb.LGBMClassifier(random_state=42, class_weight="balanced", verbose=-1)

model_all_pet_final.fit(X_train_all_pet, y_train)

print("\n Bootstrap AUC hesaplanıyor...")

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

stat, p_value = wilcoxon(auc_boot_all, auc_boot_all_pet)

print("\n Wilcoxon Signed-Rank Test")
print(f"  Test istatistiği: {stat:.3f}")
print(f"  p-değeri        : {p_value:.4f}")

if p_value < 0.05:
    print("  İstatistiksel olarak ANLAMLI fark (p < 0.05)")
else:
    print("  İstatistiksel olarak ANLAMLI fark yok (p ≥ 0.05)")

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

print("\n" + "="*80)
print(" SINIF BAZINDA DETAYLI METRİKLER")
print("="*80)

from sklearn.metrics import classification_report

def class_wise_metrics(
    y_true,
    y_pred,
    y_proba,
    class_labels,
    class_names,
    model_name
):
    metrics = []

    for i, cls in enumerate(class_labels):
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)

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

y_pred_all = model_all_final.predict(X_test_all)
y_proba_all = model_all_final.predict_proba(X_test_all)

y_pred_all_pet = model_all_pet_final.predict(X_test_all_pet)
y_proba_all_pet = model_all_pet_final.predict_proba(X_test_all_pet)

class_labels = np.unique(y_test)
class_names  = le.inverse_transform(class_labels)

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

metrics_all.to_csv("class_metrics_model_all.csv", index=False)
metrics_all_pet.to_csv("class_metrics_model_all_pet.csv", index=False)
comparison.to_csv("class_metrics_comparison.csv", index=False)

print("\n Sınıf bazlı metrik tabloları kaydedildi.")

print("\n" + "="*80)
print(" PERMUTATION IMPORTANCE ANALİZİ")
print("="*80)

from sklearn.inspection import permutation_importance

print("\n Model ALL+PET için permutation importance hesaplanıyor...")

perm_result = permutation_importance(
    model_all_pet_final,
    X_test_all_pet,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc_ovr"
)

pet_start_idx = len(all_features)
pet_feature_indices = list(
    range(pet_start_idx, pet_start_idx + len(pet_features))
)

pet_perm_importance = {
    feat: perm_result.importances_mean[idx]
    for feat, idx in zip(pet_features, pet_feature_indices)
}

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

perm_df = pd.DataFrame({
    "Feature": pet_perm_importance_sorted.keys(),
    "Permutation_Importance": pet_perm_importance_sorted.values()
})

perm_df.to_csv("pet_permutation_importance.csv", index=False)

print("\n Permutation importance tablosu kaydedildi.")

print("\n" + "="*80)
print(" CALIBRATION CURVE ANALİZİ (Sınıf Bazında)")
print("="*80)

from sklearn.calibration import calibration_curve

def plot_calibration_curve(
    y_true,
    y_proba,
    class_idx,
    class_name,
    model_label,
    ax,
    n_bins=5
):
    y_true_bin = (y_true == class_idx).astype(int)
    y_proba_cls = y_proba[:, class_idx]

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

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, cls in enumerate(class_labels):
    ax = axes[i]

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")

    plot_calibration_curve(
        y_true=y_test.values,
        y_proba=y_proba_all,
        class_idx=cls,
        class_name=class_names[i],
        model_label="Model ALL",
        ax=ax
    )

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

from sklearn.metrics import brier_score_loss

print("\n BRIER SCORE (micro-average)")

y_true_bin_all = label_binarize(y_test.values, classes=class_labels).ravel()

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

print("\n BÖLÜM tamamlandı.")
