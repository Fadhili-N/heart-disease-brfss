"""Shared experiment machinery: split, preprocessing, scoring, logging.

Every experiment, traditional or deep, goes through the same split and the same
metrics, so the rows of the results table are directly comparable.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import SEED
from src.features import BINARY_FEATURES, ENGINEERED, NOMINAL_FEATURES, NUMERIC_FEATURES

EXP_DIR = Path(__file__).resolve().parents[1] / "experiments"
RESULTS_CSV = EXP_DIR / "results.csv"
RESULTS: dict = {}          # exp_id -> row; re-running an experiment overwrites its row


def split_data(X: pd.DataFrame, y: pd.Series, seed: int = SEED) -> dict:
    """Stratified 70 / 15 / 15 split. The test set stays untouched until the end."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed)
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}


def build_preprocessor(use_engineered: bool = True, scale: bool = True,
                       extra_binary: tuple = ()) -> ColumnTransformer:
    """Imputation, missing-value flags, scaling and one-hot encoding.

    Fitted only on training rows when used inside a Pipeline. `scale=False` is for
    tree models, which do not need standardised inputs.
    """
    numeric = NUMERIC_FEATURES + (ENGINEERED if use_engineered else [])
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("bin", SimpleImputer(strategy="most_frequent", add_indicator=True),
         BINARY_FEATURES + list(extra_binary)),
        ("nom", Pipeline([("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), NOMINAL_FEATURES),
    ])


def feature_columns(use_engineered: bool = True, extra_binary: tuple = ()) -> list:
    return (NUMERIC_FEATURES + (ENGINEERED if use_engineered else []) + BINARY_FEATURES
            + list(extra_binary) + NOMINAL_FEATURES)


def best_f1_threshold(y_true, proba) -> float:
    """Probability cut-off that maximises F1; chosen on validation data only."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def score(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    p = precision_score(y_true, pred, zero_division=0)
    r = recall_score(y_true, pred)
    return {"roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba),
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if (p + r) else 0.0}


def positive_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _log_wandb(exp_id: str, config: dict, metrics: dict) -> None:
    """Write the run to a local, offline Weights & Biases log (no account needed)."""
    try:
        os.environ["WANDB_SILENT"] = "true"
        import wandb
        EXP_DIR.mkdir(exist_ok=True)
        run = wandb.init(project="heart-disease-brfss", name=exp_id, config=config,
                         dir=str(EXP_DIR), mode="offline", reinit="finish_previous")
        run.log(metrics)
        run.finish()
    except Exception as err:  # logging must never break an experiment
        print(f"[warning] wandb logging skipped for {exp_id}: {err}")


def run_experiment(exp_id: str, name: str, model, data: dict, params: str,
                   use_engineered: bool = True, scale: bool = True,
                   extra_binary: tuple = (), observation: str = "", log: bool = True) -> dict:
    """Fit a model inside the shared pipeline and record train/validation metrics."""
    cols = feature_columns(use_engineered, extra_binary)
    pipe = Pipeline([("prep", build_preprocessor(use_engineered, scale, extra_binary)),
                     ("model", model)])

    t0 = time.time()
    pipe.fit(data["X_train"][cols], data["y_train"])
    fit_seconds = time.time() - t0

    val_p = positive_proba(pipe, data["X_val"][cols])
    threshold = best_f1_threshold(data["y_val"], val_p)
    val = score(data["y_val"], val_p, threshold)

    # train metrics on a fixed 100k sample (cheap, and enough to see the train/val gap)
    sample = data["X_train"].sample(n=min(100_000, len(data["X_train"])), random_state=SEED)
    train_p = positive_proba(pipe, sample[cols])
    train = score(data["y_train"].loc[sample.index], train_p, threshold)

    row = {
        "exp_id": exp_id, "model": name, "params": params,
        "features": f"{len(cols)} ({'with' if use_engineered else 'no'} engineered)",
        "n_train": len(data["X_train"]), "n_val": len(data["X_val"]),
        "train_roc_auc": round(train["roc_auc"], 4), "val_roc_auc": round(val["roc_auc"], 4),
        "val_pr_auc": round(val["pr_auc"], 4), "threshold": round(threshold, 3),
        "precision": round(val["precision"], 4), "recall": round(val["recall"], 4),
        "f1": round(val["f1"], 4), "fit_seconds": round(fit_seconds, 1),
        "observation": observation,
    }
    RESULTS[exp_id] = row
    EXP_DIR.mkdir(exist_ok=True)
    pd.DataFrame(RESULTS.values()).to_csv(RESULTS_CSV, index=False)
    if log:
        _log_wandb(exp_id, {k: row[k] for k in ("model", "params", "features", "n_train", "n_val")},
                   {k: row[k] for k in row if k.startswith(("train_", "val_")) or k in
                    ("threshold", "precision", "recall", "f1", "fit_seconds")})
    return {"row": row, "pipeline": pipe, "val_proba": val_p, "threshold": threshold}


def results_table() -> pd.DataFrame:
    return pd.DataFrame(RESULTS.values()).set_index("exp_id")
