# Preoperative Prediction of Residual Cancer Burden After Neoadjuvant Chemotherapy in Breast Cancer

Analysis code accompanying the manuscript *"Preoperative Prediction of Residual Cancer Burden After Neoadjuvant Chemotherapy in Breast Cancer: A Multimodal Machine Learning Approach and Implications for Clinical Decision Support."*

The code was written for a doctoral thesis and its comments, variable names, and printed output are in Turkish. This README provides an English map of the pipeline, a Turkish–English variable glossary, and a table linking each script to the figures and tables in the manuscript.

---

## 1. Study in brief

Four-class residual cancer burden (RCB-0 / RCB-I / RCB-II / RCB-III) is predicted from data available **before surgery** in a single-centre retrospective cohort of 328 patients treated with neoadjuvant chemotherapy (NAC).

Candidate predictors are grouped into six clinically defined blocks, and eleven configurations are evaluated along a stepwise pipeline. The central comparison is between a radiology-only model (**Model R**, 17 features) and a fully integrated multimodal model (**Model ALL**, 62 features).

> **Block-label note.** In the code, the comorbidity block is labelled `K` (from Turkish *komorbidite*). In the manuscript this block is labelled **C**. `Model K` in the code corresponds to **Model C** in the paper, `Model P+O+D+K` to **Model P+O+D+C**, and `Model P+O+D+K+B` to **Model P+O+D+C+B**.

| Block | Code label | Paper label | Features | Domain |
|---|---|---|---|---|
| Pathologic | P | P | 11 | Receptor status, subtype, Ki-67, grading, TILs |
| Oncologic | O | O | 5 | Metastasis, stage, regimen, cycle intensity |
| Demographic | D | D | 5 | Affected breast, BMI class, age group, blood type, sun exposure |
| Comorbidity | **K** | **C** | 10 | Hypertension, diabetes, COPD, smoking, family history, etc. |
| Biochemical | B | B | 14 | ALP, ALT, AST, BUN, CA 15-3, CEA, CRP, GGT, glucose, HbA1c, creatinine, LDH, TSH, eGFR |
| Radiologic | R | R | 17 | BI-RADS, breast density, lesion type, mass and calcification descriptors, etc. |

---

## 2. Repository layout

```
Codes/
  ML (3.Bölüm)/
    Modüler ve Kademeli RCB Sınıflandırması ile Makine Öğrenmesi (Clean).py
    Modüler ve Kademeli RCB Sınıflandırması ile Makine Öğrenmesi.py
    Modüler ve Kademeli RCB Sınıflandırması ile Makine Öğrenmesi.docx
    Model ALL İçin Geliştirilen Doğrulama Analizi (Clean).py
    Model ALL İçin Geliştirilen Doğrulama Analizi.py
    Model ALL İçin Geliştirilen Doğrulama Analizi.docx
Results/
  Modüler Makine Öğrenmesi/
  Doğrulama Analizleri/
```

The `(Clean).py` files are the tidied versions and are the recommended entry points. The `.docx` files are annotated walkthroughs of the same code.

| File | English name | What it does |
|---|---|---|
| `Modüler ve Kademeli RCB Sınıflandırması ile Makine Öğrenmesi (Clean).py` | Modular and stepwise RCB classification | Builds the six blocks, runs the full grid of 11 configurations × 3 algorithms × 2 SMOTE states (66 runs), and reports test AUC, macro-F1, accuracy, and the cross-validation-to-test gap |
| `Model ALL İçin Geliştirilen Doğrulama Analizi (Clean).py` | Validation analysis for Model ALL | Runs 5×5 nested cross-validation, 500 bootstrap resamples for class-wise confidence intervals, and TreeSHAP global and per-class explainability for the selected final model |

---

## 3. Mapping to the manuscript

| Manuscript item | Produced by |
|---|---|
| Table 2 — core performance indicators | Modular script (`Model R` and `Model ALL` rows of the grid) |
| Figure 1 — test AUC across 11 configurations | Modular script |
| Figure 2 — Model R vs Model ALL key metrics | Modular script |
| Figure 3 — class-wise F1 and AUC | Modular script |
| Supplementary Table S1 — best configuration per model | Modular script (full 66-run grid) |
| Supplementary Table S2 — synergy analysis | Modular script (single-block AUCs) |
| Supplementary Table S3 — class-wise test performance | Modular script |
| Supplementary Table S4 — nested CV and bootstrap | Validation script |
| Supplementary Table S5 — algorithm-invariance grid | Modular script (Model R and Model ALL rows, all algorithms and SMOTE states) |
| Supplementary Figures S1–S6 — confusion matrix, ROC, PR, calibration, cumulative gain, lift | Validation script |
| Supplementary Figures S7, S9–S12 — global and per-class SHAP | Validation script |
| Supplementary Figure S8 — nested CV gap across outer folds | Validation script |

---

## 4. Environment

Python 3.11–3.12.

```
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
imbalanced-learn==0.11.0
shap==0.42.0
pandas==2.0.3
numpy==1.24.0
statsmodels==0.14.0
matplotlib
seaborn
```

A fixed seed (`random_state=42`) is used throughout.

Run the modular script first, then the validation script.

---

## 5. Data availability

**No patient-level data are included in this repository.** The analyses require a single tabular dataset with one row per patient, the 62 predictor columns, and the RCB outcome column.

The anonymised dataset is available from the corresponding author on reasonable request, subject to restrictions arising from patient privacy and the terms of the ethics approval (Non-Interventional Clinical Research Ethics Committee, approval no. 2024/25-08, 17 July 2024).

To reproduce the analyses, place the dataset at the path expected by the scripts and adjust the file path constant at the top of each script.

---

## 6. Turkish–English variable glossary

**Radiologic (Model R)**

| Turkish | English |
|---|---|
| BI-RADS | BI-RADS |
| Meme Dansitesi | Breast density |
| Lokalizasyon | Localization |
| Lezyon Türü | Lesion type |
| Mimari (Distorsiyon) | Architectural distortion |
| Kitle Şekli | Mass shape |
| Kitle Konturu | Mass margin |
| Kitle Dansitesi | Mass density |
| Kalsifikasyon Morfolojisi | Calcification morphology |
| Kalsifikasyon Dağılımı | Calcification distribution |
| Asimetri | Asymmetry |
| Multifokalite | Multifocality |
| 2 Yıldır Stabil | Lesion stable for 2 years |
| Deri Retraksiyonu | Skin retraction |
| Meme Başı Retraksiyonu | Nipple retraction |
| Ameliyat Öyküsü | Prior surgery |
| Kozmetik İmplant | Cosmetic implant |

**Pathologic and oncologic**

| Turkish | English |
|---|---|
| Histolojik Tip | Histologic type |
| Moleküler Tip | Molecular subtype |
| Mitotik Derece | Mitotic grade |
| Nükleer Derece | Nuclear grade |
| Tübül Derecesi | Tubule grade |
| Histolojik Derece | Histologic grade |
| Metastaz Yeri | Metastasis site |
| Tanı Evresi | Stage at diagnosis |
| Rejim | Regimen |
| Kür Yoğunluğu | Cycle intensity |

**Demographic, comorbidity, biochemical**

| Turkish | English |
|---|---|
| Yaş Grubu | Age group |
| VKİ Sınıfı | BMI class |
| Etkilenen Meme | Affected breast |
| Kan Grubu | Blood type |
| Güneş Maruziyeti | Sun exposure |
| Hipertansiyon | Hypertension |
| Diyabet | Diabetes mellitus |
| KOAH | COPD |
| Sigara | Smoking |
| Ailede Meme Kanseri Öyküsü | Family history of breast cancer |
| Tiroid Hastalığı | Thyroid disease |
| Retinopati | Retinopathy |
| Nöropati | Neuropathy |
| Osteoporoz | Osteoporosis |
| Depresyon | Depression |
| Glukoz | Glucose |
| Kreatinin | Creatinine |
| e-GFR | eGFR |

**Recurring output terms**

| Turkish | English |
|---|---|
| Modüler ve Kademeli | Modular and stepwise |
| Doğrulama Analizi | Validation analysis |
| SMOTE Yok / SMOTE Var | SMOTE not applied / SMOTE applied |
| Özellik | Feature |
| Eğitim / Test | Training / Test |
| Karışıklık Matrisi | Confusion matrix |
| Kalibrasyon | Calibration |
| Duyarlılık / Özgüllük | Recall / Specificity |
| Kazanç Eğrisi | Cumulative gain curve |
| Sonuçlar | Results |

---

## 7. Citation

If you use this code, please cite the accompanying article and this repository.

---

## 8. Licence

Released under the MIT Licence. See [`LICENSE`](LICENSE).
