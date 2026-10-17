"""Keras / tf.data side of the project.

Uses the same split, the same preprocessing and the same metrics as the
scikit-learn experiments, so rows of the results table stay comparable.
"""
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, regularizers
from sklearn.model_selection import train_test_split

from src.data import SEED
from src.experiment import (EXP_DIR, RESULTS, RESULTS_CSV, best_f1_threshold,
                            build_preprocessor, feature_columns, score)

HISTORIES: dict = {}        # exp_id -> per-epoch history, used for learning curves


# ---------------------------------------------------------------- input pipeline
def _dense(a) -> np.ndarray:
    return (a.toarray() if hasattr(a, "toarray") else np.asarray(a)).astype("float32")


def make_inputs(data: dict, use_engineered: bool = False, stop_fraction: float = 0.10) -> dict:
    """Fit the preprocessor on training rows and carve out an early-stopping slice.

    The slice comes from the training data, so the validation set stays untouched
    by training decisions, as with early stopping in the boosting experiments.
    """
    cols = feature_columns(use_engineered)
    prep = build_preprocessor(use_engineered, scale=True)
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        data["X_train"][cols], data["y_train"], test_size=stop_fraction,
        stratify=data["y_train"], random_state=SEED)
    out = {
        "prep": prep, "n_features": None,
        "X_fit": _dense(prep.fit_transform(X_fit)), "y_fit": y_fit.to_numpy("float32"),
        "X_stop": _dense(prep.transform(X_stop)), "y_stop": y_stop.to_numpy("float32"),
        "X_val": _dense(prep.transform(data["X_val"][cols])), "y_val": data["y_val"].to_numpy("float32"),
        "X_test": _dense(prep.transform(data["X_test"][cols])), "y_test": data["y_test"].to_numpy("float32"),
        "use_engineered": use_engineered,
    }
    out["n_features"] = out["X_fit"].shape[1]
    return out


def make_dataset(X, y, batch_size: int = 1024, shuffle: bool = False) -> tf.data.Dataset:
    """tf.data pipeline: tensors -> (shuffle) -> batch -> prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- architectures
def sequential_mlp(n_features: int, units=(64, 32), dropout: float = 0.0) -> keras.Model:
    """Plain stack of dense layers built with the Sequential API."""
    model = keras.Sequential([layers.Input((n_features,))])
    for u in units:
        model.add(layers.Dense(u, activation="relu"))
        if dropout:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def functional_mlp(n_features: int, units=(128, 64, 32), dropout: float = 0.3,
                   l2: float = 0.0, batch_norm: bool = True) -> keras.Model:
    """Same idea built with the Functional API, with batch norm, dropout and L2 as options."""
    inp = layers.Input((n_features,))
    x = inp
    for u in units:
        x = layers.Dense(u, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))


# ---------------------------------------------------------------- experiment runner
def balanced_class_weight(y) -> dict:
    n, pos = len(y), float(np.sum(y))
    return {0: n / (2 * (n - pos)), 1: n / (2 * pos)}


def run_keras_experiment(exp_id: str, name: str, model_fn, inputs: dict, params: str,
                         learning_rate: float = 1e-3, batch_size: int = 1024, max_epochs: int = 40,
                         patience: int = 5, class_weight=None, observation: str = "") -> dict:
    """Train a Keras model, then score it exactly like the scikit-learn runs."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = model_fn(inputs["n_features"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(name="pr_auc", curve="PR")])

    train_ds = make_dataset(inputs["X_fit"], inputs["y_fit"], batch_size, shuffle=True)
    stop_ds = make_dataset(inputs["X_stop"], inputs["y_stop"], batch_size)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=stop_ds, epochs=max_epochs, shuffle=False,
                     class_weight=class_weight, callbacks=callbacks, verbose=0)
    fit_seconds = time.time() - t0
    epochs_run = len(hist.history["loss"])
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    HISTORIES[exp_id] = hist.history

    val_p = model.predict(make_dataset(inputs["X_val"], inputs["y_val"], 4096), verbose=0).ravel()
    threshold = best_f1_threshold(inputs["y_val"], val_p)
    val = score(inputs["y_val"], val_p, threshold)

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(inputs["X_fit"]), size=min(100_000, len(inputs["X_fit"])), replace=False)
    tr_p = model.predict(make_dataset(inputs["X_fit"][idx], inputs["y_fit"][idx], 4096), verbose=0).ravel()
    train = score(inputs["y_fit"][idx], tr_p, threshold)

    row = {
        "exp_id": exp_id, "model": name,
        "params": f"{params}; best epoch {best_epoch} of {epochs_run}",
        "features": f"{inputs['n_features']} model inputs ({'with' if inputs['use_engineered'] else 'no'} engineered)",
        "n_train": len(inputs["X_fit"]), "n_val": len(inputs["X_val"]),
        "train_roc_auc": round(train["roc_auc"], 4), "val_roc_auc": round(val["roc_auc"], 4),
        "val_pr_auc": round(val["pr_auc"], 4), "threshold": round(threshold, 3),
        "precision": round(val["precision"], 4), "recall": round(val["recall"], 4),
        "f1": round(val["f1"], 4), "fit_seconds": round(fit_seconds, 1),
        "observation": observation,
    }
    RESULTS[exp_id] = row
    EXP_DIR.mkdir(exist_ok=True)
    pd.DataFrame(RESULTS.values()).to_csv(RESULTS_CSV, index=False)
    _log_wandb_history(exp_id, row, hist.history)
    return {"row": row, "model": model, "val_proba": val_p, "threshold": threshold}


def _log_wandb_history(exp_id: str, row: dict, history: dict) -> None:
    """Offline W&B log with one entry per epoch, so learning curves are traceable."""
    try:
        os.environ["WANDB_SILENT"] = "true"
        import wandb
        run = wandb.init(project="heart-disease-brfss", name=exp_id, dir=str(EXP_DIR), mode="offline",
                         reinit="finish_previous",
                         config={k: row[k] for k in ("model", "params", "features", "n_train", "n_val")})
        for epoch in range(len(history["loss"])):
            run.log({k: v[epoch] for k, v in history.items()}, step=epoch)
        run.summary.update({k: row[k] for k in row if k.startswith(("train_", "val_")) or k in
                            ("threshold", "precision", "recall", "f1", "fit_seconds")})
        run.finish()
    except Exception as err:
        print(f"[warning] wandb logging skipped for {exp_id}: {err}")


# ---------------------------------------------------------------- learning curves
def plot_learning_curves(exp_ids, path=None):
    """Training versus early-stopping loss for each run, side by side."""
    fig, axes = plt.subplots(1, len(exp_ids), figsize=(4.2 * len(exp_ids), 3.4), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, eid in zip(axes, exp_ids):
        h = HISTORIES[eid]
        epochs = np.arange(1, len(h["loss"]) + 1)
        ax.plot(epochs, h["loss"], label="train", color="#4c72b0")
        ax.plot(epochs, h["val_loss"], label="held-out slice", color="#c44e52")
        ax.axvline(int(np.argmin(h["val_loss"])) + 1, color="grey", ls=":", lw=1)
        ax.set_title(eid)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("binary cross-entropy")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()
