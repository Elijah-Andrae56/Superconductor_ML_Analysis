
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


RANDOM_SEED = 42


def fit_pca(X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int = 2):
    """Fit PCA on the training set and transform both train and test sets."""
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X_train.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return X_train_pca, X_test_pca, pca, loadings


def fit_kmeans(X_train: pd.DataFrame, k: int = 3):
    """Fit K-Means to training data and return model and labels."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X_train)
    return kmeans, labels


def build_random_forest():
    """Create the Random Forest regressor used in the analysis."""
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def evaluate_regression(y_true, y_pred) -> dict:
    """Return standard regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def fit_evaluate_random_forest(X_train, X_test, y_train, y_test):
    """Train the Random Forest and evaluate on a held-out test set."""
    model = build_random_forest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    return model, y_pred, metrics


def cross_validate_random_forest(X, y, cv: int = 5):
    """Run cross-validation and return summary statistics."""
    model = build_random_forest()
    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, scoring="r2", cv=kfold, n_jobs=-1)
    return {
        "cv_r2_mean": float(scores.mean()),
        "cv_r2_std": float(scores.std()),
        "cv_r2_scores": scores.tolist(),
    }


def feature_importance_table(model, feature_names, top_n: int = 15):
    """Return a dataframe of feature importances sorted from largest to smallest."""
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importances.head(top_n)
