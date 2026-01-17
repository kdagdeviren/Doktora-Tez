import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, f1_score,
    roc_curve, auc
)

from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline  # IMPORTANT: SMOTE inside CV folds

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    has_lgbm = True
except Exception as e:
    has_lgbm = False
    print(" LightGBM import edilemedi. Kurmak için: !pip install lightgbm")

try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception as e:
    has_xgb = False
    print(" XGBoost import edilemedi. Kurmak için: !pip install xgboost")

plt.rcParams['figure.figsize'] = (12, 8)

plt.rcParams['font.size'] = 12

sns.set_style("whitegrid")

RANDOM_STATE = 42

N_SPLITS = 5

print("=== PET VERİLERİ İLE RCB SINIFLANDIRMA (RF + LGBM + XGB) + ŞEKİLLER ===")

print("Lütfen Excel dosyanızı yükleyin:")

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

data = pd.read_excel(file_name)

print(f"Veri yüklendi: {data.shape}")

pet_features = [
    'SUVmax', 'SUVmean4', 'TLG', 'MTV', 'Yüzey/Hacim Oranı4', 'Küresellik4',
    'Asferisite4', 'SUV Varyansı4', 'SUV Eğriliği4', 'GLCM Entropi4',
    'GLCM Kontrast4', 'GLRLM Non-Uniformite4', 'NGTDM Coarseness4', 'GLSZM Entropi4'
]

all_features = [
    'i1','i2','i3','i4','i5','i6','i7','i8','i9','i10','i12',
    'i13','i14','i15','i46','i47',
    'i16','i17','i18','i19','i45',
    'i21','i22','i23','i24','i25','i26','i27','i28','i29','i30',
    'i31','i32','i33','i34','i35','i36','i37','i38','i39','i40','i41','i42','i43','i44',
    'i48','i49','i50','i51','i52','i53','i54','i55','i56','i57','i58','i59','i60','i61','i62','i63','i64'
]

target = 'RCB_Kategorize'

print(f"\n=== PET VERİSİ OLAN HASTALAR ===")

pet_data = data.dropna(subset=pet_features)

print(f"PET verisi olan hasta sayısı: {len(pet_data)}")

print(f"\n=== KATEGORİK DEĞİŞKENLERİ SAYISAL KODLARA ÇEVİRME ===")

pet_data_encoded = pet_data.copy()

for col in pet_features:
    if pet_data_encoded[col].dtype == 'object':
        print(f"'{col}' kategorik → sayısal kodlama")
        pet_data_encoded[col] = pet_data_encoded[col].astype('category').cat.codes
    else:
        print(f"'{col}' zaten sayısal")

pet_data_encoded[target] = pet_data_encoded[target].astype('category').cat.codes

n_classes = pet_data_encoded[target].nunique()
print(f"\nSınıf sayısı: {n_classes} (0..{n_classes-1})")

models = {
    'Model PET': pet_features,
    'Model ALL': all_features,
    'Model ALL + PET': all_features + pet_features
}

print(f"\n=== 3 MODEL TANIMI ===")
for model_name, feats in models.items():
    print(f"{model_name}: {len(feats)} özellik")

algorithms = {}

algorithms["RandomForest"] = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)

if has_lgbm:
    algorithms["LightGBM"] = LGBMClassifier(
        random_state=RANDOM_STATE,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9
    )

if has_xgb:
    algorithms["XGBoost"] = XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=-1
    )

print("\n=== KULLANILACAK ALGORİTMALAR ===")
for k in algorithms.keys():
    print("-", k)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "auc": "roc_auc_ovr",
    "acc": "accuracy",
    "f1": "f1_macro"
}

results = []

best_test_artifacts = {}

print(
    f"\n=== ANALİZ BAŞLIYOR (3 MODEL × {len(algorithms)} ALGORİTMA × SMOTE'lu/SMOTE'suz) ==="
)

from sklearn.base import clone

for model_name, feats in models.items():
    print(f"\n==============================")
    print(f"MODEL: {model_name} ({len(feats)} özellik)")
    print(f"==============================")

    X = pet_data_encoded[feats]

    y = pet_data_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    for algo_name, clf_template in algorithms.items():
        print(f"\n--- Algorithm: {algo_name} ---")

        clf_no = clone(clf_template)

        cv_no = cross_validate(
            clf_no,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        clf_no.fit(X_train, y_train)

        y_pred_no = clf_no.predict(X_test)

        y_proba_no = clf_no.predict_proba(X_test)

        test_auc_no = roc_auc_score(
            y_test,
            y_proba_no,
            multi_class="ovr",
            average="macro"
        )

        test_acc_no = accuracy_score(y_test, y_pred_no)
        test_f1_no = f1_score(y_test, y_pred_no, average="macro")

        results.append({
            "Model": model_name,
            "Algorithm": algo_name,
            "SMOTE_Durumu": "SMOTE'suz",
            "CV_AUC_Mean": np.mean(cv_no["test_auc"]),
            "CV_AUC_Std":  np.std(cv_no["test_auc"], ddof=0),
            "CV_Accuracy_Mean": np.mean(cv_no["test_acc"]),
            "CV_Accuracy_Std":  np.std(cv_no["test_acc"], ddof=0),
            "CV_F1_Mean": np.mean(cv_no["test_f1"]),
            "CV_F1_Std":  np.std(cv_no["test_f1"], ddof=0),
            "Test_AUC": test_auc_no,
            "Test_Accuracy": test_acc_no,
            "Test_F1": test_f1_no
        })

        clf_sm = clone(clf_template)

        smote = SMOTE(random_state=RANDOM_STATE)

        pipe = Pipeline([
            ("smote", smote),
            ("clf", clf_sm)
        ])

        cv_yes = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        pipe.fit(X_train, y_train)

        y_pred_yes = pipe.predict(X_test)
        y_proba_yes = pipe.predict_proba(X_test)

        test_auc_yes = roc_auc_score(
            y_test,
            y_proba_yes,
            multi_class="ovr",
            average="macro"
        )
        test_acc_yes = accuracy_score(y_test, y_pred_yes)
        test_f1_yes = f1_score(y_test, y_pred_yes, average="macro")

        results.append({
            "Model": model_name,
            "Algorithm": algo_name,
            "SMOTE_Durumu": "SMOTE'lu",
            "CV_AUC_Mean": np.mean(cv_yes["test_auc"]),
            "CV_AUC_Std":  np.std(cv_yes["test_auc"], ddof=0),
            "CV_Accuracy_Mean": np.mean(cv_yes["test_acc"]),
            "CV_Accuracy_Std":  np.std(cv_yes["test_acc"], ddof=0),
            "CV_F1_Mean": np.mean(cv_yes["test_f1"]),
            "CV_F1_Std":  np.std(cv_yes["test_f1"], ddof=0),
            "Test_AUC": test_auc_yes,
            "Test_Accuracy": test_acc_yes,
            "Test_F1": test_f1_yes
        })

        best_test_artifacts[(model_name, algo_name)] = {
            "y_test": y_test,
            "y_pred": y_pred_yes,
            "y_proba": y_proba_yes,
            "fitted": pipe
        }

        print(
            f"SMOTE'suz  | CV AUC: {np.mean(cv_no['test_auc']):.3f}±{np.std(cv_no['test_auc']):.3f} "
            f"| Test AUC: {test_auc_no:.3f}"
        )
        print(
            f"SMOTE'lu   | CV AUC: {np.mean(cv_yes['test_auc']):.3f}±{np.std(cv_yes['test_auc']):.3f} "
            f"| Test AUC: {test_auc_yes:.3f}"
        )

results_df = pd.DataFrame(results)

print("\n=== TÜM SONUÇLAR (ÖZET) ===")
print(
    results_df
    .sort_values(["Model", "Algorithm", "SMOTE_Durumu"])
    .to_string(index=False)
)

with pd.ExcelWriter("PET_Analiz_Sonuclari_TUM_ALG.xlsx", engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Results_Long", index=False)

    pivot_auc = results_df.pivot_table(
        index=["Model", "Algorithm"],
        columns="SMOTE_Durumu",
        values=[
            "CV_AUC_Mean", "CV_AUC_Std",
            "Test_AUC", "Test_Accuracy", "Test_F1"
        ],
        aggfunc="first"
    )

    pivot_auc.to_excel(writer, sheet_name="Results_Pivot")

print("\n Sonuçlar 'PET_Analiz_Sonuclari_TUM_ALG.xlsx' dosyasına kaydedildi!")

print("\n=== ŞEKİLLER ÜRETİLİYOR ===")

plt.figure(figsize=(14, 6))

smote_yes_df = results_df[results_df["SMOTE_Durumu"] == "SMOTE'lu"].copy()

sns.barplot(
    data=smote_yes_df,
    x="Model",
    y="Test_AUC",
    hue="Algorithm"
)

plt.title("Test AUC Karşılaştırması (SMOTE'lu) - Tüm Algoritmalar")
plt.ylim(0, 1)

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("PET_TestAUC_SMOTEli_TUM_ALG.png", dpi=300, bbox_inches="tight")

plt.show()

colors = ["blue", "red", "green"]

model_order = list(models.keys())

for algo_name in algorithms.keys():
    plt.figure(figsize=(12, 8))

    for i, mname in enumerate(model_order):
        art = best_test_artifacts[(mname, algo_name)]

        y_test = art["y_test"]
        y_proba = art["y_proba"]

        classes = list(range(n_classes))

        y_bin = label_binarize(y_test, classes=classes)

        fpr, tpr = {}, {}

        for c in classes:
            fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], y_proba[:, c])

        all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))

        mean_tpr = np.zeros_like(all_fpr)

        for c in classes:
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])

        mean_tpr /= len(classes)

        mean_auc = auc(all_fpr, mean_tpr)

        plt.plot(
            all_fpr,
            mean_tpr,
            color=colors[i],
            lw=2,
            label=f"{mname} (macro AUC={mean_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Eğrileri (SMOTE'lu Test) - {algo_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"PET_ROC_SMOTEli_{algo_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

for algo_name in algorithms.keys():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, mname in enumerate(model_order):
        art = best_test_artifacts[(mname, algo_name)]
        y_test = art["y_test"]
        y_pred = art["y_pred"]

        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["RCB-0", "RCB-I", "RCB-II", "RCB-III"][:n_classes],
            yticklabels=["RCB-0", "RCB-I", "RCB-II", "RCB-III"][:n_classes]
        )

        axes[i].set_title(f"{mname}\n{algo_name} (SMOTE'lu)")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"PET_CM_SMOTEli_{algo_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

print("\n Şekiller üretildi:")
print("- PET_TestAUC_SMOTEli_TUM_ALG.png")
for algo_name in algorithms.keys():
    print(f"- PET_ROC_SMOTEli_{algo_name}.png")
    print(f"- PET_CM_SMOTEli_{algo_name}.png")

files.download("PET_Analiz_Sonuclari_TUM_ALG.xlsx")
files.download("PET_TestAUC_SMOTEli_TUM_ALG.png")

for algo_name in algorithms.keys():
    files.download(f"PET_ROC_SMOTEli_{algo_name}.png")
    files.download(f"PET_CM_SMOTEli_{algo_name}.png")

print("\n Dosyalar indirildi.")
