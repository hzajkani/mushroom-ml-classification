# 🍄 Mushroom Classification — Asymmetric Cost ML Competition

A machine learning project that classifies mushrooms as **poisonous** or **edible** with a focus on **zero False Negatives** — because eating a poisonous mushroom is not a mistake you get to make twice.

## The Problem

In asymmetric classification, not all errors are equal. Here:

- **False Negative** (poisonous → predicted edible) = ☠️ catastrophic
- **False Positive** (edible → predicted poisonous) = 🤷 just a missed meal

The goal is to maximize the number of edible mushrooms we can safely eat while **never** misclassifying a poisonous one.

## Approach

A single Random Forest achieves ~96% accuracy but assigns **0.0 probability** to one poisonous outlier — no threshold can catch it. The solution is a **3-model ensemble**:

| Model | Role |
|-------|------|
| Random Forest (500 trees) | High overall accuracy |
| Logistic Regression (balanced) | Catches the outlier (assigns it ~0.87 prob) |
| Gradient Boosting (500 trees) | Additional decision boundary diversity |

Averaged probabilities + a **cost-balanced tuned threshold (~0.25)** yields:

| Metric | Baseline RF | Ensemble (Final) |
|--------|-------------|-------------------|
| Accuracy | 0.96 | 0.93 |
| Precision | 0.95 | 0.88 |
| Recall | 0.97 | **1.00** |
| False Negatives | ~79 | **0** ✅ |

## Project Structure

```
mushroom-ml-classification/
├── README.md
├── mushroom_competition.ipynb      # Full analysis notebook
├── mushroom_submission.csv         # Competition submission file
└── data/
    ├── 7.4.3.1_mushroom_competition_train_data.csv
    └── 7.4.3.2_mushroom_competition_test_data.csv
```

## Features Used

All categorical features from a subset of the [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom):

`cap.shape` · `cap.color` · `bruises` · `stalk.color.above.ring` · `stalk.color.below.ring` · `population`

## Key Techniques

- **OneHotEncoding** for categorical feature preprocessing
- **10-Fold Cross-Validation** throughout
- **ROC & Precision-Recall curve** analysis
- **Cost-balanced scoring** (FN cost = 100× FP cost)
- **Ensemble averaging** to eliminate blind spots of individual models
- **Threshold tuning** with `TunedThresholdClassifierCV` and manual grid search

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## Usage

```bash
jupyter notebook mushroom_competition.ipynb
```

Make sure the CSV data files are in a `data/` subdirectory relative to the notebook.

## Context

Built as part of the **WBS Coding School — Data Science & AI** program, Module 7: Asymmetric Classification.
