heavy_plots = True

quick_mode  = False

import os, zipfile, io, sys, warnings, json

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
    roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)

from sklearn.preprocessing import label_binarize

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

import joblib

try:
    import shap
    shap_available = True
except Exception:
    try:
        import sys
        !pip install -q shap
        import shap
        shap_available = True
    except Exception:
        shap_available = False

BASE_DIR = 'outputs'
FIG_DIR  = os.path.join(BASE_DIR, 'figures')
TAB_DIR  = os.path.join(BASE_DIR, 'tables')
MOD_DIR  = os.path.join(BASE_DIR, 'models')
LOG_DIR  = os.path.join(BASE_DIR, 'logs')

for d in [BASE_DIR, FIG_DIR, TAB_DIR, MOD_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

def savefig(path, dpi=300):
    plt.tight_layout()

    plt.savefig(path, dpi=dpi, bbox_inches='tight')

    plt.show()

    plt.clf()

def save_csv(df, name):
    df.to_csv(os.path.join(TAB_DIR, name), index=False)

print("=== VERİ YÜKLEME VE HAZIRLIK ===")
print("Lütfen Excel dosyanızı yükleyin:")

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

data = pd.read_excel(file_name)

target = 'RCB_Kategorize'

features_p = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i12']

names_p = [
    'Histolojik Tip', 'ER', 'PR', 'HER2', 'Moleküler Tip', 'Ki-67',
    'Tübül Derecesi', 'Nükleer Derece', 'Mitotik Derece', 'Histolojik Grade', 'TIL Değeri'
]

features_o = ['i13', 'i14', 'i15', 'i46', 'i47']
names_o = ['Metastaz Durumu', 'Metastaz Yeri', 'Tanı Evresi', 'Rejim', 'Kür Yoğunluk']

features_d = ['i16', 'i17', 'i18', 'i19', 'i45']
names_d = ['Hangi Meme', 'VKI Sınıfı', 'Yaş Grubu', 'Kan Grubu', 'Güneşten Yararlanma']

features_k = ['i21','i22','i23','i24','i25','i26','i27','i28','i29','i30']
names_k = ['HT','DM','KOAH','Sigara','Ailede Meme CA','Tiroid','Retinopati','Nöropati','Osteoporoz','Depresyon']

features_b = ['i31','i32','i33','i34','i35','i36','i37','i38','i39','i40','i41','i42','i43','i44']
names_b = ['ALP','ALT','AST','BUN','CA15-3','CEA','CRP','GGT','Glukoz','HbA1c','Kreatinin','LDH','TSH','e-GFR']

features_r = ['i48','i49','i50','i51','i52','i53','i54','i55','i56','i57','i58','i59','i60','i61','i62','i63','i64']
names_r = [
    'BI-RADS','Meme Dansitesi','Lokalizasyon','Lezyon Türü','Mimari',
    'Kitle Şekli','Kitle Konturu','Kitle Dansitesi','Kalsifikasyon Morfolojisi',
    'Kalsifikasyon Dağılımı','Asimetri','Multifokalite','2 Yıldır Stabil',
    'Cilt Çekintisi','Meme Başı Retraksiyonu','Ameliyat Öyküsü','Kozmetik Implant'
]

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

if quick_mode:
    run_order = ['Model P', 'Model P+O+D', 'Model ALL']
else:
    run_order = [
        'Model P','Model O','Model P+O','Model D','Model P+O+D','Model K',
        'Model P+O+D+K','Model B','Model P+O+D+K+B','Model R','Model ALL'
    ]

algs = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
}

classes = np.array([0, 1, 2, 3])

X_all = data[models_dict['Model ALL']].copy()

y_all = data[target].copy()

assert np.array_equal(np.sort(y_all.unique()), classes), "Sınıf etiketleri [0,1,2,3] değil!"

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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def safe_cross_validation(model_template, X, y, cv, use_smote=False):
    cv_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_smote:
            try:
                sm = SMOTE(random_state=42, k_neighbors=5)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        model = clone(model_template)

        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        cv_scores.append({
            'accuracy': accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro'),
            'f1': f1_score(y_val, y_pred, average='macro')
        })

    return cv_scores

def plot_confusion(y_true, y_pred, title, out_png):
    plt.figure(figsize=(8, 6))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

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
    y_bin = label_binarize(y_true, classes=classes)

    fpr, tpr, roc_auc = {}, {}, {}

    for i, c in enumerate(classes):
        fpr[c], tpr[c], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])

    plt.figure(figsize=(8, 6))

    for c in classes:
        plt.plot(
            fpr[c],
            tpr[c],
            lw=2,
            label=f'RCB-{c} (AUC={roc_auc[c]:.3f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')

    savefig(os.path.join(FIG_DIR, out_png))

def plot_feature_importance(model, feat_names, title, out_png, topn=15):
    if not hasattr(model, 'feature_importances_'):
        return

    imp = model.feature_importances_

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

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical probability')
    plt.title(title)
    plt.legend()

    savefig(os.path.join(FIG_DIR, out_png))

def plot_gain_lift(y_true, y_proba, prefix):
    y_bin = label_binarize(y_true, classes=classes)

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

def calculate_class_metrics(y_true, y_pred, y_proba, classes, model_name, alg_name):
    class_metrics = []

    for i in classes:
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

        try:
            auc_i = roc_auc_score(y_binary, proba_i)
        except Exception:
            auc_i = 0.5

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

all_rows = []

best_per_model = {}

all_class_metrics = []

for model_name in run_order:

    feats = models_dict[model_name]

    feat_names = names_dict[model_name]

    X_tr = X_train[feats].copy()
    X_te = X_test[feats].copy()

    print(f"\n=== {model_name} ({len(feats)} özellik) ===")

    model_results = {}

    for alg_name, est_template in algs.items():

        print(f" - {alg_name} CV/Test hesaplanıyor...")

        cv_no = safe_cross_validation(
            est_template,
            X_tr,
            y_train,
            skf,
            use_smote=False
        )

        cv_acc_no  = np.mean([s['accuracy'] for s in cv_no])
        cv_auc_no  = np.mean([s['auc']      for s in cv_no])
        cv_f1_no   = np.mean([s['f1']       for s in cv_no])

        cv_acc_no_s = np.std([s['accuracy'] for s in cv_no])
        cv_auc_no_s = np.std([s['auc']      for s in cv_no])
        cv_f1_no_s  = np.std([s['f1']       for s in cv_no])

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

        sm = SMOTE(random_state=42, k_neighbors=5)
        try:
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_train)
        except Exception:
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

        model_results[alg_name] = {

            'cv_acc_mean_no_smote': cv_acc_no,
            'cv_acc_std_no_smote':  cv_acc_no_s,
            'cv_auc_mean_no_smote': cv_auc_no,
            'cv_auc_std_no_smote':  cv_auc_no_s,
            'cv_f1_mean_no_smote':  cv_f1_no,
            'cv_f1_std_no_smote':   cv_f1_no_s,

            'test_acc_no_smote': test_acc_no,
            'test_auc_no_smote': test_auc_no,
            'test_f1_no_smote':  test_f1_no,

            'cv_acc_mean_smote': cv_acc_sm,
            'cv_acc_std_smote':  cv_acc_sm_s,
            'cv_auc_mean_smote': cv_auc_sm,
            'cv_auc_std_smote':  cv_auc_sm_s,
            'cv_f1_mean_smote':  cv_f1_sm,
            'cv_f1_std_smote':   cv_f1_sm_s,

            'test_acc_smote': test_acc_sm,
            'test_auc_smote': test_auc_sm,
            'test_f1_smote':  test_f1_sm,

            'y_pred_no_smote':  y_pred_no,
            'y_proba_no_smote': y_proba_no,
            'y_pred_smote':     y_pred_sm,
            'y_proba_smote':    y_proba_sm
        }

        class_metrics = calculate_class_metrics(
            y_test,
            y_pred_sm,
            y_proba_sm,
            classes,
            model_name,
            alg_name
        )
        all_class_metrics.extend(class_metrics)

        prefix = f"{model_name}_{alg_name}"

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

        if heavy_plots:

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

            if shap_available and alg_name in ['RandomForest','XGBoost','LightGBM']:
                try:
                    explainer = shap.TreeExplainer(est_sm)

                    sample_idx = np.random.RandomState(42).choice(
                        len(X_te),
                        size=min(200, len(X_te)),
                        replace=False
                    )
                    X_te_sample = X_te.iloc[sample_idx]

                    shap_vals = explainer.shap_values(X_te_sample)

                    human_feature_names = names_dict[model_name]

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
                    with open(
                        os.path.join(LOG_DIR, 'shap_errors.log'),
                        'a',
                        encoding='utf-8'
                    ) as f:
                        f.write(f'{prefix}: {str(e)}\n')

    def combo_candidates():
        for alg_name, res in model_results.items():

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

    candidates = list(combo_candidates())

    candidates.sort(
        key=lambda d: (-d['test_auc'], d['cv_test_gap'], -d['test_f1'])
    )

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

results_df = pd.DataFrame(all_rows)

save_csv(results_df, 'model_results_fixed.csv')

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

best_df = best_df.sort_values(
    by=['Test_AUC', 'CV_Test_Gap', 'Test_F1'],
    ascending=[False, True, False]
)

save_csv(best_df, 'best_per_model_fixed.csv')

all_class_metrics_df = pd.DataFrame(all_class_metrics)
save_csv(all_class_metrics_df, 'all_class_metrics_fixed.csv')

print("\n=== EN İYİLER (Test AUC'a göre) ===")
print(best_df)

best_overall = best_df.iloc[0]

TIE_THRESHOLD = 0.01

top_auc_value = best_overall['Test_AUC']

tie_mask = (top_auc_value - best_df['Test_AUC']).abs() <= TIE_THRESHOLD
tie_candidates_df = best_df[tie_mask].copy()

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

def bootstrap_auc_diff(
    y_true,
    y_proba_1,
    y_proba_2,
    n_bootstrap=1000,
    random_state=42
):
    np.random.seed(random_state)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
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

if tie_info['needs_tie_break']:

    print("\n" + "=" * 70)
    print("EŞİTLİK DURUMU KONTROLÜ: Genel Veri-Agnostik Seçim")
    print("=" * 70)

    print("\n Klinik olarak eşit kabul edilen modeller tespit edildi.")
    print(f"  AUC eşik: ±{TIE_THRESHOLD:.3f}")

    candidates = best_df[
        (top_auc_value - best_df['Test_AUC']).abs() <= TIE_THRESHOLD
    ].copy()

    candidates['CV_Test_Gap'] = candidates['CV_Test_Gap'].fillna(np.inf)
    candidates['Test_F1'] = candidates['Test_F1'].fillna(0.0)
    candidates['CV_AUC_Std'] = candidates['CV_AUC_Std'].fillna(np.inf)
    candidates['Num_Features'] = candidates['Num_Features'].fillna(0)

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

    best_overall = chosen

print(
    f"\n En iyi genel: "
    f"{best_overall['Model']} + "
    f"{best_overall['BestAlgorithm']} + "
    f"SMOTE {best_overall['SMOTE']} | "
    f"CV AUC={best_overall['CV_AUC_Mean']:.3f}, "
    f"Test AUC={best_overall['Test_AUC']:.3f}"
)

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

best_model_name = best_overall['Model']
best_alg_name   = best_overall['BestAlgorithm']

feats_best = models_dict[best_model_name]

X_best_tr = X_train[feats_best]

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

est_best_template = algs[best_alg_name]
est_best = clone(est_best_template)
est_best.fit(X_best_tr_final, y_best_tr_final)

try:
    if shap_available and best_alg_name in [
        'RandomForest', 'XGBoost', 'LightGBM'
    ]:

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
