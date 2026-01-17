import os, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, brier_score_loss
)

from sklearn.preprocessing import label_binarize

from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

import joblib

print("=== VERİ YÜKLEME ===")
print("Lütfen Excel dosyanızı yükleyin:")

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

data = pd.read_excel(file_name)

target = 'RCB_Kategorize'

classes = np.array([0, 1, 2, 3])

features_p = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i12']
features_o = ['i13', 'i14', 'i15', 'i46', 'i47']
features_d = ['i16', 'i17', 'i18', 'i19', 'i45']
features_k = ['i21','i22','i23','i24','i25','i26','i27','i28','i29','i30']
features_b = ['i31','i32','i33','i34','i35','i36','i37','i38','i39','i40','i41','i42','i43','i44']
features_r = ['i48','i49','i50','i51','i52','i53','i54','i55','i56','i57','i58','i59','i60','i61','i62','i63','i64']

feats_all = features_p + features_o + features_d + features_k + features_b + features_r

X_all = data[feats_all].copy()

y_all = data[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    stratify=y_all,
    random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Sınıf oranları (train/test):")
print(y_train.value_counts(normalize=True).sort_index())
print(y_test.value_counts(normalize=True).sort_index())

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

model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)

print("\n=== NESTED CV ANALİZİ ===")
print("Dış loop: 5-fold, İç loop: 5-fold (toplam 25 model)")
print("Bu, overfitting kontrolü için daha katı bir değerlendirme sağlar.\n")

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

nested_scores = {
    'outer_fold': [],
    'class': [],
    'inner_cv_auc_mean': [],
    'inner_cv_auc_std': [],
    'outer_test_auc': [],
    'gap': []
}

for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X_train, y_train)
):
    X_tr_outer = X_train.iloc[train_idx]
    X_te_outer = X_train.iloc[test_idx]
    y_tr_outer = y_train.iloc[train_idx]
    y_te_outer = y_train.iloc[test_idx]

    inner_aucs_per_class = {c: [] for c in classes}

    for inner_train_idx, inner_val_idx in inner_cv.split(
        X_tr_outer, y_tr_outer
    ):
        X_tr_inner = X_tr_outer.iloc[inner_train_idx]
        X_val_inner = X_tr_outer.iloc[inner_val_idx]
        y_tr_inner = y_tr_outer.iloc[inner_train_idx]
        y_val_inner = y_tr_outer.iloc[inner_val_idx]

        est_inner = clone(model)
        est_inner.fit(X_tr_inner, y_tr_inner)

        y_proba_inner = est_inner.predict_proba(X_val_inner)

        y_bin_inner = label_binarize(
            y_val_inner,
            classes=classes
        )

        for i, c in enumerate(classes):
            try:
                auc_c = roc_auc_score(
                    y_bin_inner[:, i],
                    y_proba_inner[:, i]
                )
            except:
                auc_c = 0.5

            inner_aucs_per_class[c].append(auc_c)

    est_outer = clone(model)
    est_outer.fit(X_tr_outer, y_tr_outer)

    y_proba_outer = est_outer.predict_proba(X_te_outer)

    y_bin_outer = label_binarize(
        y_te_outer,
        classes=classes
    )

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

        gap = inner_mean - outer_auc

        nested_scores['outer_fold'].append(outer_fold)
        nested_scores['class'].append(f'RCB-{c}')
        nested_scores['inner_cv_auc_mean'].append(inner_mean)
        nested_scores['inner_cv_auc_std'].append(inner_std)
        nested_scores['outer_test_auc'].append(outer_auc)
        nested_scores['gap'].append(gap)

nested_df = pd.DataFrame(nested_scores)

print("\nNested CV Sonuçları (Sınıf Bazlı):")
print(
    nested_df.groupby('class').agg({
        'inner_cv_auc_mean': ['mean', 'std'],
        'outer_test_auc': ['mean', 'std'],
        'gap': ['mean', 'std']
    }).round(3)
)

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

print("\n=== SINIF BAZLI BOOTSTRAP CI'LARI ===")
print("Her RCB sınıfı için ayrı bootstrap CI (500 tekrar, %95)\n")

def bootstrap_class_metrics(
    y_true,
    y_pred,
    y_proba,
    class_idx,
    n_bootstrap=500,
    random_state=42
):
    np.random.seed(random_state)

    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values

    n = len(y_true)

    metrics_list = {
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
        'specificity': []
    }

    y_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    proba_class = y_proba[:, class_idx]

    for _ in range(n_bootstrap):

        idx = np.random.choice(n, size=n, replace=True)

        y_true_boot = y_binary[idx]
        y_pred_boot = y_pred_binary[idx]
        proba_boot = proba_class[idx]

        tp = np.sum((y_true_boot == 1) & (y_pred_boot == 1))
        fp = np.sum((y_true_boot == 0) & (y_pred_boot == 1))
        fn = np.sum((y_true_boot == 1) & (y_pred_boot == 0))
        tn = np.sum((y_true_boot == 0) & (y_pred_boot == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        try:
            auc_val = roc_auc_score(y_true_boot, proba_boot)
        except:
            auc_val = 0.5

        metrics_list['precision'].append(precision)
        metrics_list['recall'].append(recall)
        metrics_list['f1'].append(f1)
        metrics_list['auc'].append(auc_val)
        metrics_list['specificity'].append(specificity)

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

    for metric_name, stats in results.items():
        print(
            f"  {metric_name.capitalize()}: "
            f"{stats['mean']:.3f} "
            f"(%95 CI: {stats['ci_lower']:.3f} - {stats['ci_upper']:.3f}, "
            f"medyan: {stats['median']:.3f})"
        )

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

print("\n=== DECISION CURVE ANALYSIS ===")
print("Her RCB sınıfı için DCA eğrisi oluşturuluyor...\n")

def calculate_net_benefit(y_true, y_proba, threshold):
    n = len(y_true)

    y_pred = (y_proba >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if threshold == 0 or threshold == 1:
        return 0.0

    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit

def calculate_treat_all_net_benefit(y_true, threshold):
    n = len(y_true)

    tp = np.sum(y_true == 1)
    fp = n - tp

    if threshold == 0 or threshold == 1:
        return 0.0

    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit

thresholds = np.arange(0.1, 0.81, 0.05)

fig_dca, ax_dca = plt.subplots(figsize=(12, 8))

colors = ['blue', 'orange', 'green', 'red']
class_names = ['RCB-0', 'RCB-1', 'RCB-2', 'RCB-3']

for class_idx, (c, color, name) in enumerate(zip(classes, colors, class_names)):

    y_true_binary = (y_test.values == c).astype(int)
    y_proba_binary = y_proba_test[:, class_idx]

    net_benefits = []

    for pt in thresholds:
        nb = calculate_net_benefit(
            y_true_binary,
            y_proba_binary,
            pt
        )
        net_benefits.append(nb)

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

ax_dca.axhline(
    y=0,
    color='black',
    linestyle='-',
    linewidth=2,
    label='Treat-none'
)

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

print("\n=== CALIBRATION CURVE ANALYSIS ===")
print("Raw, Platt ve Isotonic kalibrasyon yöntemleri uygulanıyor...\n")

y_true_binary_all = label_binarize(
    y_test,
    classes=classes
)

y_proba_all = y_proba_test

y_true_micro = y_true_binary_all.ravel()
y_proba_micro = y_proba_all.ravel()

fraction_of_positives_raw, mean_predicted_value_raw = calibration_curve(
    y_true_micro,
    y_proba_micro,
    n_bins=10,
    strategy='uniform'
)

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

fig_cal, ax_cal = plt.subplots(figsize=(10, 8))

ax_cal.plot(
    [0, 1],
    [0, 1],
    'k--',
    label='Perfect Calibration',
    linewidth=2
)

ax_cal.plot(
    mean_predicted_value_raw,
    fraction_of_positives_raw,
    'o-',
    color='blue',
    linewidth=2,
    markersize=8,
    label=f'Raw (Brier={brier_raw:.3f})'
)

ax_cal.plot(
    mean_predicted_value_platt,
    fraction_of_positives_platt,
    's--',
    color='orange',
    linewidth=2,
    markersize=6,
    label=f'Platt (Brier={brier_platt:.3f})'
)

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
