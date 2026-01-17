import json
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

from google.colab import files

print("=" * 80)
print("ÇOKLU LOJİSTİK REGRESYON - ADIM 3: MODEL KURULUMU VE DEĞERLENDİRME")
print("=" * 80)
print()

print("1. VERİ YÜKLEME")
print("-" * 80)

print("Lütfen Excel dosyanızı yükleyin:")
uploaded = files.upload()
excel_path = list(uploaded.keys())[0]

data = pd.read_excel(excel_path)
print(f"Veri yüklendi: {data.shape[0]} hasta, {data.shape[1]} sütun\n")

try:
    with open("vif_summary_report.json", "r", encoding="utf-8") as f:
        vif_summary = json.load(f)
    with open("dummy_coding_info.json", "r", encoding="utf-8") as f:
        dummy_info = json.load(f)
    print("ADIM 2 sonuçları yüklendi (vif_summary_report.json + dummy_coding_info.json)\n")
except FileNotFoundError:
    print(" HATA: ADIM 2 sonuçları bulunamadı!")
    print("   Lütfen önce ADIM 2'yi çalıştırın ve çıktı dosyalarının aynı çalışma dizininde olduğundan emin olun.")
    raise

vif1 = pd.read_csv("model1_vif_results.csv")
vif2 = pd.read_csv("model2_vif_results.csv")

target = "RCB_ML"

if target not in data.columns:
    raise ValueError(f"HATA: '{target}' sütunu veri setinde yok!")

y = data[target].astype(int)

if y.nunique() < 2:
    raise ValueError(" HATA: RCB_ML tek sınıf içeriyor. Lojistik regresyon için 0 ve 1 olmalı!")

print(f"RCB_ML sınıf dağılımı:\n{y.value_counts()}\n")

print("2. DUMMY DEĞİŞKENLERİ OLUŞTURMA")
print("-" * 80)

def create_dummy_variables(df_in, var_dict, ref_dict):
    X_dummy = pd.DataFrame(index=df_in.index)

    for var_code, var_name in var_dict.items():
        if var_code not in df_in.columns:
            continue

        categories = sorted(df_in[var_code].dropna().unique())
        if len(categories) == 0:
            continue

        ref_cat = ref_dict.get(var_code, categories[0])

        dummy_cols = pd.get_dummies(
            df_in[var_code],
            prefix=var_code,
            drop_first=False
        ).astype(int)

        ref_col = f"{var_code}_{ref_cat}"
        if ref_col in dummy_cols.columns:
            dummy_cols = dummy_cols.drop(columns=[ref_col])

        rename_map = {}
        for col in dummy_cols.columns:
            parts = col.split("_", 1)
            cat_part = parts[1] if len(parts) > 1 else "NA"
            rename_map[col] = f"{var_code}_cat{cat_part}"
        dummy_cols = dummy_cols.rename(columns=rename_map)

        for cat in categories:
            if cat == ref_cat:
                continue
            col_name = f"{var_code}_cat{cat}"
            if col_name not in dummy_cols.columns:
                dummy_cols[col_name] = 0

        dummy_cols = dummy_cols[sorted(dummy_cols.columns)]

        X_dummy = pd.concat([X_dummy, dummy_cols], axis=1)

    return X_dummy.astype(float)

model1_vars_dict = {k: v["name"] for k, v in dummy_info["model1"].items()}
model2_vars_dict = {k: v["name"] for k, v in dummy_info["model2"].items()}
reference_categories = dummy_info["reference_categories"]

X1_dummy = create_dummy_variables(data, model1_vars_dict, reference_categories)
X2_dummy = create_dummy_variables(data, model2_vars_dict, reference_categories)

print(f"Model 1: {X1_dummy.shape[1]} dummy değişken oluşturuldu")
print(f"Model 2: {X2_dummy.shape[1]} dummy değişken oluşturuldu\n")

print("3. MULTICOLLINEARITY DÜZELTMESİ")
print("-" * 80)

def remove_multicollinearity(X_dummy, vif_df, correlation_threshold=0.7, vif_threshold=5.0):
    X_clean = X_dummy.copy()

    perfect_cols = vif_df.loc[vif_df["VIF"] >= 999.0, "Değişken"].tolist()
    perfect_cols = [c for c in perfect_cols if c in X_clean.columns]

    if len(perfect_cols) > 0:
        print(f"Perfect multicollinearity (VIF>=999): {len(perfect_cols)} kolon çıkarılıyor")
        for c in perfect_cols:
            print(f"   - {c}")
        X_clean = X_clean.drop(columns=perfect_cols)
    else:
        print(" Perfect multicollinearity yok (VIF>=999 bulunmadı)")

    high_vif_cols = vif_df.loc[
        (vif_df["VIF"] >= vif_threshold) & (vif_df["VIF"] < 999.0),
        "Değişken"
    ].tolist()
    high_vif_cols = [c for c in high_vif_cols if c in X_clean.columns]

    removed_due_to_corr = []

    if len(high_vif_cols) > 1:
        corr_matrix = X_clean[high_vif_cols].corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                v1 = corr_matrix.columns[i]
                v2 = corr_matrix.columns[j]
                r = corr_matrix.iloc[i, j]

                if abs(r) > correlation_threshold:
                    vif1_val = float(vif_df.loc[vif_df["Değişken"] == v1, "VIF"].values[0])
                    vif2_val = float(vif_df.loc[vif_df["Değişken"] == v2, "VIF"].values[0])

                    var_to_remove = v1 if vif1_val >= vif2_val else v2

                    if (var_to_remove in X_clean.columns) and (var_to_remove not in removed_due_to_corr):
                        removed_due_to_corr.append(var_to_remove)
                        print(f"  |r|>{correlation_threshold} (r={abs(r):.3f}) → {var_to_remove} çıkarılıyor (VIF daha yüksek)")
                        X_clean = X_clean.drop(columns=[var_to_remove])

        if len(removed_due_to_corr) == 0:
            print(f" Yüksek korelasyonlu (|r|>{correlation_threshold}) çift bulunmadı")
    else:
        print("  Korelasyon taraması için yeterli yüksek-VIF değişken yok")

    return X_clean

print("Model 1 için düzeltme:")
X1_clean = remove_multicollinearity(X1_dummy, vif1, correlation_threshold=0.7, vif_threshold=5.0)
print(f" Model 1: {X1_clean.shape[1]} değişken kaldı\n")

print("Model 2 için düzeltme:")
X2_clean = remove_multicollinearity(X2_dummy, vif2, correlation_threshold=0.7, vif_threshold=5.0)
print(f" Model 2: {X2_clean.shape[1]} değişken kaldı\n")

print("4. TRAIN-TEST SPLIT")
print("-" * 80)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1_clean,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2_clean,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Model 1:")
print(f"  Train: {X1_train.shape[0]} hasta, {X1_train.shape[1]} değişken")
print(f"  Test : {X1_test.shape[0]} hasta, {X1_test.shape[1]} değişken")
print(f"  Train pCR oranı: {y1_train.mean():.3f}")
print(f"  Test  pCR oranı: {y1_test.mean():.3f}\n")

print("Model 2:")
print(f"  Train: {X2_train.shape[0]} hasta, {X2_train.shape[1]} değişken")
print(f"  Test : {X2_test.shape[0]} hasta, {X2_test.shape[1]} değişken")
print(f"  Train pCR oranı: {y2_train.mean():.3f}")
print(f"  Test  pCR oranı: {y2_test.mean():.3f}\n")

print("5. MODEL KURULUMU VE DEĞERLENDİRME")
print("-" * 80)

def evaluate_model(
    X_train, X_test, y_train, y_test,
    model_name, regularization,
    C=None,
    cv_splits=5,
    correlation_threshold=0.7
):
    if regularization == "l1":
        penalty = "l1"
        solver = "liblinear"
    elif regularization == "l2":
        penalty = "l2"
        solver = "lbfgs"
    else:
        raise ValueError("regularization sadece 'l1' veya 'l2' olabilir")

    best_C = C

    if best_C is None:
        param_grid = {
            "C": [0.001, 0.01, 0.05, 0.07, 0.1, 0.144, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        grid = GridSearchCV(
            LogisticRegression(
                penalty=penalty,
                solver=solver,
                max_iter=2000,
                random_state=42
            ),
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_C = grid.best_params_["C"]

    model = LogisticRegression(
        penalty=penalty,
        C=best_C,
        solver=solver,
        max_iter=2000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_pred_proba_test >= 0.5).astype(int)

    cm_default = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm_default.ravel()

    sens_default = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_default = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc_default = accuracy_score(y_test, y_pred_default)
    ppv_default = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv_default = tn / (tn + fn) if (tn + fn) > 0 else 0

    auc_test = roc_auc_score(y_test, y_pred_proba_test)

    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_pred_proba_train)
    auc_gap = auc_train - auc_test

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = []

    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        cv_model = LogisticRegression(
            penalty=penalty,
            C=best_C,
            solver=solver,
            max_iter=2000,
            random_state=42
        )
        cv_model.fit(X_tr, y_tr)
        val_proba = cv_model.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, val_proba))

    cv_auc_mean = float(np.mean(cv_scores))
    cv_auc_std = float(np.std(cv_scores))

    fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba_train)

    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    optimal_threshold = float(thresholds[best_idx])

    y_pred_opt = (y_pred_proba_test >= optimal_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    tn2, fp2, fn2, tp2 = cm_opt.ravel()

    sens_opt = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    spec_opt = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0
    acc_opt = accuracy_score(y_test, y_pred_opt)
    ppv_opt = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
    npv_opt = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0

    y_train_pred_opt = (y_pred_proba_train >= optimal_threshold).astype(int)
    cm_train_opt = confusion_matrix(y_train, y_train_pred_opt)
    tn3, fp3, fn3, tp3 = cm_train_opt.ravel()

    sens_train_opt = tp3 / (tp3 + fn3) if (tp3 + fn3) > 0 else 0
    spec_train_opt = tn3 / (tn3 + fp3) if (tn3 + fp3) > 0 else 0

    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    feature_names = X_train.columns.tolist()

    results = {
        "model_name": model_name,
        "regularization": regularization,
        "C": float(best_C),

        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,

        "train_auc": float(auc_train),
        "test_auc": float(auc_test),
        "auc_gap": float(auc_gap),

        "test_accuracy_default": float(acc_default),
        "sensitivity_default": float(sens_default),
        "specificity_default": float(spec_default),
        "ppv_default": float(ppv_default),
        "npv_default": float(npv_default),
        "confusion_matrix_default": cm_default.tolist(),

        "optimal_threshold": optimal_threshold,
        "test_accuracy": float(acc_opt),
        "sensitivity": float(sens_opt),
        "specificity": float(spec_opt),
        "ppv": float(ppv_opt),
        "npv": float(npv_opt),
        "confusion_matrix": cm_opt.tolist(),

        "sensitivity_train_opt": float(sens_train_opt),
        "specificity_train_opt": float(spec_train_opt),

        "intercept": float(intercept),
        "coefficients": {n: float(c) for n, c in zip(feature_names, coefs)},

        "y_test": y_test.tolist(),
        "y_pred": y_pred_opt.tolist(),
        "y_pred_proba": y_pred_proba_test.tolist(),
    }

    return results, model

all_results = []

print("Model 1 - L1 Regularization (Lasso):")
r1_l1, m1_l1 = evaluate_model(X1_train, X1_test, y1_train, y1_test, "Model 1", "l1")
all_results.append(r1_l1)
print(f"  Optimal C: {r1_l1['C']:.3f}")
print(f"  Optimal Eşik (Youden): {r1_l1['optimal_threshold']:.3f}")
print(f"  CV AUC: {r1_l1['cv_auc_mean']:.3f} ± {r1_l1['cv_auc_std']:.3f}")
print(f"  Train AUC: {r1_l1['train_auc']:.3f}")
print(f"  Test AUC: {r1_l1['test_auc']:.3f}")
print(f"  AUC Gap: {r1_l1['auc_gap']:.3f}")
print(f"  Test Acc (Opt): {r1_l1['test_accuracy']:.3f}")
print(f"  Sens (Opt): {r1_l1['sensitivity']:.3f}")
print(f"  Spec (Opt): {r1_l1['specificity']:.3f}\n")

print("Model 1 - L2 Regularization (Ridge):")
r1_l2, m1_l2 = evaluate_model(X1_train, X1_test, y1_train, y1_test, "Model 1", "l2")
all_results.append(r1_l2)
print(f"  Optimal C: {r1_l2['C']:.3f}")
print(f"  Optimal Eşik (Youden): {r1_l2['optimal_threshold']:.3f}")
print(f"  CV AUC: {r1_l2['cv_auc_mean']:.3f} ± {r1_l2['cv_auc_std']:.3f}")
print(f"  Train AUC: {r1_l2['train_auc']:.3f}")
print(f"  Test AUC: {r1_l2['test_auc']:.3f}")
print(f"  AUC Gap: {r1_l2['auc_gap']:.3f}")
print(f"  Test Acc (Opt): {r1_l2['test_accuracy']:.3f}")
print(f"  Sens (Opt): {r1_l2['sensitivity']:.3f}")
print(f"  Spec (Opt): {r1_l2['specificity']:.3f}\n")

print("Model 2 - L1 Regularization (Lasso):")
r2_l1, m2_l1 = evaluate_model(X2_train, X2_test, y2_train, y2_test, "Model 2", "l1")
all_results.append(r2_l1)
print(f"  Optimal C: {r2_l1['C']:.3f}")
print(f"  Optimal Eşik (Youden): {r2_l1['optimal_threshold']:.3f}")
print(f"  CV AUC: {r2_l1['cv_auc_mean']:.3f} ± {r2_l1['cv_auc_std']:.3f}")
print(f"  Train AUC: {r2_l1['train_auc']:.3f}")
print(f"  Test AUC: {r2_l1['test_auc']:.3f}")
print(f"  AUC Gap: {r2_l1['auc_gap']:.3f}")
print(f"  Test Acc (Opt): {r2_l1['test_accuracy']:.3f}")
print(f"  Sens (Opt): {r2_l1['sensitivity']:.3f}")
print(f"  Spec (Opt): {r2_l1['specificity']:.3f}\n")

print("Model 2 - L2 Regularization (Ridge):")
r2_l2, m2_l2 = evaluate_model(X2_train, X2_test, y2_train, y2_test, "Model 2", "l2")
all_results.append(r2_l2)
print(f"  Optimal C: {r2_l2['C']:.3f}")
print(f"  Optimal Eşik (Youden): {r2_l2['optimal_threshold']:.3f}")
print(f"  CV AUC: {r2_l2['cv_auc_mean']:.3f} ± {r2_l2['cv_auc_std']:.3f}")
print(f"  Train AUC: {r2_l2['train_auc']:.3f}")
print(f"  Test AUC: {r2_l2['test_auc']:.3f}")
print(f"  AUC Gap: {r2_l2['auc_gap']:.3f}")
print(f"  Test Acc (Opt): {r2_l2['test_accuracy']:.3f}")
print(f"  Sens (Opt): {r2_l2['sensitivity']:.3f}")
print(f"  Spec (Opt): {r2_l2['specificity']:.3f}\n")

print("6. MODEL KARŞILAŞTIRMASI VE EN İYİ MODEL SEÇİMİ")
print("-" * 80)

results_df = pd.DataFrame([
    {
        "Model": r["model_name"],
        "Regularization": r["regularization"],
        "C": r["C"],
        "Optimal Eşik": f"{r['optimal_threshold']:.3f}",
        "CV AUC (Mean±Std)": f"{r['cv_auc_mean']:.3f}±{r['cv_auc_std']:.3f}",
        "Train AUC": f"{r['train_auc']:.3f}",
        "Test AUC": f"{r['test_auc']:.3f}",
        "AUC Gap": f"{r['auc_gap']:.3f}",
        "Test Accuracy": f"{r['test_accuracy']:.3f}",
        "Sensitivity": f"{r['sensitivity']:.3f}",
        "Specificity": f"{r['specificity']:.3f}",
        "PPV": f"{r['ppv']:.3f}",
        "NPV": f"{r['npv']:.3f}",
    }
    for r in all_results
])

print("\nTÜM MODELLERİN PERFORMANS KARŞILAŞTIRMASI:")
print(results_df.to_string(index=False))

best_idx = None
best_auc = -1
best_gap = 999

for i, r in enumerate(all_results):
    if r["sensitivity"] == 0.0:
        print(f"  {r['model_name']} {r['regularization']}: Sensitivity=0 → ATLANDI")
        continue
    if r["auc_gap"] > 0.15:
        continue

    if (r["test_auc"] > best_auc) or (r["test_auc"] == best_auc and r["auc_gap"] < best_gap):
        best_auc = r["test_auc"]
        best_gap = r["auc_gap"]
        best_idx = i

if best_idx is None:
    print("\n  Filtrelerden geçen model yok (Sensitivity=0 veya overfitting yüksek).")
    candidates = [(i, r) for i, r in enumerate(all_results) if r["sensitivity"] > 0]
    if len(candidates) > 0:
        best_idx = max(candidates, key=lambda t: t[1]["test_auc"])[0]
    else:
        best_idx = 0

best_model = all_results[best_idx]

print("\n EN İYİ MODEL (Optimal Eşik ile):")
print(f"  Model: {best_model['model_name']}")
print(f"  Regularization: {best_model['regularization']}")
print(f"  Optimal C: {best_model['C']:.3f}")
print(f"  Optimal Eşik (Youden): {best_model['optimal_threshold']:.3f}")
print(f"  Test AUC: {best_model['test_auc']:.3f}")
print(f"  AUC Gap: {best_model['auc_gap']:.3f}")
print(f"  Test Accuracy: {best_model['test_accuracy']:.3f}")
print(f"  Sensitivity: {best_model['sensitivity']:.3f}")
print(f"  Specificity: {best_model['specificity']:.3f}")
print(f"  PPV: {best_model['ppv']:.3f}")
print(f"  NPV: {best_model['npv']:.3f}")

print("\n   Eşik Karşılaştırması (Default 0.5 vs Optimal):")
print(f"     Default (0.5): Sens={best_model['sensitivity_default']:.3f}, Spec={best_model['specificity_default']:.3f}")
print(f"     Optimal ({best_model['optimal_threshold']:.3f}): Sens={best_model['sensitivity']:.3f}, Spec={best_model['specificity']:.3f}")

print("\n7. SONUÇLARI KAYDETME")
print("-" * 80)

with open("model_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print("Model sonuçları kaydedildi: model_results.json")

results_df.to_excel("model_comparison.xlsx", index=False)
print("Model karşılaştırması kaydedildi: model_comparison.xlsx")

best_coefs_df = pd.DataFrame({
    "Değişken": list(best_model["coefficients"].keys()),
    "Beta_Katsayı": list(best_model["coefficients"].values()),
})
best_coefs_df["OR"] = np.exp(best_coefs_df["Beta_Katsayı"])

best_coefs_df = best_coefs_df.sort_values(
    by="Beta_Katsayı",
    key=lambda s: s.abs(),
    ascending=False
)

best_coefs_df.to_excel("best_model_coefficients.xlsx", index=False)
print("En iyi model katsayıları kaydedildi: best_model_coefficients.xlsx")

files.download("model_results.json")
files.download("model_comparison.xlsx")
files.download("best_model_coefficients.xlsx")

print(f"\n{'='*80}")
print(" ADIM 3 TAMAMLANDI!")
print(f"{'='*80}")
