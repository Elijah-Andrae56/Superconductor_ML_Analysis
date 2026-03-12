# Machine Learning Prediction of Superconducting Critical Temperature

## Overview

This project applies unsupervised and supervised machine learning to the UCI Superconductivity dataset in order to study whether compositional descriptors can predict superconducting critical temperature (Tc).

The analysis is framed as a materials informatics workflow rather than a class exercise. It combines:

- exploratory data analysis
- feature standardization
- principal component analysis (PCA)
- K-Means clustering
- Random Forest regression
- cross-validation and model evaluation
- feature importance analysis

The goal is not only to predict Tc accurately, but also to examine whether physically meaningful structure emerges in the feature space.

## Why this project matters

Discovering new superconductors is difficult because candidate materials occupy a large, high-dimensional compositional space. Machine learning provides a practical way to screen that space, identify broad material families, and estimate which descriptors are most predictive of superconducting behavior.

This project demonstrates an end-to-end ML workflow that is relevant to:
- materials informatics
- condensed matter and superconductivity
- scientific machine learning
- data-driven discovery in physics and engineering

## Dataset

- **Source:** UCI Machine Learning Repository, Superconductivity Dataset
- **Observations:** ~21,000 superconducting compounds
- **Target:** `critical_temp`
- **Features:** 81 numerical descriptors derived from elemental properties such as atomic mass, radius, valence, density, electron affinity, and thermal conductivity

Place the dataset CSV at:

```text
data/superconductor.csv
```

## Methods

### 1. Preprocessing
Features are split into train and test sets and standardized using `StandardScaler`.

### 2. Dimensionality Reduction
PCA is used to project the 81-dimensional feature space into two principal components for visualization. The model is fit on the training set only and then applied to the test set to avoid leakage.

### 3. Unsupervised Learning
K-Means clustering is applied in the standardized feature space to investigate whether materials form natural groups associated with different critical temperature regimes.

### 4. Supervised Learning
A Random Forest regressor is trained to predict critical temperature from the compositional descriptors.

### 5. Evaluation
Performance is measured using:
- R²
- RMSE
- MAE
- 5-fold cross-validation

### 6. Interpretation
Feature importances are used to identify which physical descriptors contribute most strongly to predictive performance.

## Key Improvements Over the Original Class Notebook

This repository refactors the original notebook into a cleaner and more credible analysis by:
- fixing preprocessing and PCA leakage issues
- using reusable helper functions
- separating code into `src/`
- improving scientific framing and markdown explanations
- adding cross-validation
- adding feature importance analysis
- organizing the project as a portfolio-ready repository

## Repository Structure

```text
superconductor-portfolio/
├── README.md
├── requirements.txt
├── data/
│   └── superconductor.csv
├── notebooks/
│   └── portfolio_analysis.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── plots.py
│   └── utils.py
└── figures/
```

## Main Results

When run on the original project setup, the Random Forest model produced approximately:

- **R²:** 0.93
- **RMSE:** 8.99 K
- **MAE:** 5.09 K

These values indicate that compositional descriptors contain substantial information about superconducting critical temperature, although the model should still be treated as a predictive screening tool rather than a mechanistic theory.

## Future Extensions

The following extensions would make the project even stronger:
- compare tree ensembles with gradient boosting
- investigate log-transformed or stratified target analysis
- cluster materials first, then fit cluster-specific regressors
- integrate SHAP values for richer interpretability
- compare composition-only features with crystal-structure-aware descriptors

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/portfolio_analysis.ipynb
```
