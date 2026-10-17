"""Final-evaluation helpers: uncertainty, curves, calibration, subgroups, operating points."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from src.data import SEED
from src.experiment import score

COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def _boot_indices(n: int, n_boot: int, seed: int = SEED):
    rng = np.random.RandomState(seed)
    return [rng.randint(0, n, n) for _ in range(n_boot)]


def bootstrap_ci(y, p, n_boot: int = 300) -> dict:
    """95% percentile intervals for ROC-AUC and PR-AUC by resampling test people."""
    y, p = np.asarray(y), np.asarray(p)
    roc, pr = [], []
    for idx in _boot_indices(len(y), n_boot):
        roc.append(roc_auc_score(y[idx], p[idx]))
        pr.append(average_precision_score(y[idx], p[idx]))
    f = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return {"roc_lo": f(roc)[0], "roc_hi": f(roc)[1], "pr_lo": f(pr)[0], "pr_hi": f(pr)[1]}


def paired_bootstrap(y, p_a, p_b, n_boot: int = 300) -> dict:
    """Difference in AUC between two models on the *same* resampled people (a minus b)."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    d_roc, d_pr = [], []
    for idx in _boot_indices(len(y), n_boot):
        d_roc.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
        d_pr.append(average_precision_score(y[idx], p_a[idx]) - average_precision_score(y[idx], p_b[idx]))
    out = {}
    for name, d in (("roc", d_roc), ("pr", d_pr)):
        out[f"{name}_diff"] = float(np.mean(d))
        out[f"{name}_lo"], out[f"{name}_hi"] = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return out


def test_summary(y, probas: dict, thresholds: dict, n_boot: int = 300) -> pd.DataFrame:
    rows = []
    for name, p in probas.items():
        s = score(y, p, thresholds[name])
        ci = bootstrap_ci(y, p, n_boot)
        rows.append({
            "model": name,
            "ROC-AUC": f"{s['roc_auc']:.3f} [{ci['roc_lo']:.3f}, {ci['roc_hi']:.3f}]",
            "PR-AUC": f"{s['pr_auc']:.3f} [{ci['pr_lo']:.3f}, {ci['pr_hi']:.3f}]",
            "Brier": round(brier_score_loss(y, p), 4),
            "threshold": round(thresholds[name], 3),
            "precision": round(s["precision"], 3), "recall": round(s["recall"], 3), "F1": round(s["f1"], 3),
        })
    return pd.DataFrame(rows).set_index("model")


def plot_roc_pr(y, probas: dict, path=None):
    y = np.asarray(y)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for (name, p), c in zip(probas.items(), COLORS):
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(fpr, tpr, color=c, label=f"{name} ({roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        b.plot(rec, prec, color=c, label=f"{name} ({average_precision_score(y, p):.3f})")
    a.plot([0, 1], [0, 1], ls=":", color="grey")
    a.set(xlabel="false positive rate", ylabel="true positive rate (recall)", title="ROC curve (test set)")
    b.axhline(y.mean(), ls=":", color="grey", label=f"no skill ({y.mean():.3f})")
    b.set(xlabel="recall", ylabel="precision", title="Precision-recall curve (test set)")
    a.legend(frameon=False, loc="lower right"); b.legend(frameon=False)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


def plot_confusion(y, probas: dict, thresholds: dict, path=None):
    fig, axes = plt.subplots(1, len(probas), figsize=(4.2 * len(probas), 3.8))
    for ax, (name, p) in zip(np.atleast_1d(axes), probas.items()):
        cm = confusion_matrix(y, (p >= thresholds[name]).astype(int))
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.0%} of row)", ha="center", va="center",
                        color="white" if pct[i, j] > 0.5 else "black", fontsize=9)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no disease", "disease"],
               yticklabels=["no disease", "disease"], xlabel="predicted", ylabel="actual", title=name)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


def plot_calibration(y, probas: dict, path=None):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for (name, p), c in zip(probas.items(), COLORS):
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", color=c, label=name)
    top = max(0.6, ax.get_xlim()[1])
    ax.plot([0, top], [0, top], ls=":", color="grey", label="perfectly calibrated")
    ax.set(xlabel="mean predicted probability", ylabel="observed fraction with heart disease",
           title="Calibration (test set, 10 quantile bins)")
    ax.legend(frameon=False)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


def operating_points(y_val, p_val, y_test, p_test, targets=(0.5, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    """Choose a threshold on validation for each recall target, then measure it on test."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    y_test, p_test = np.asarray(y_test), np.asarray(p_test)
    rows = []
    for t in targets:
        ok = np.where(rec[:-1] >= t)[0]
        cut = thr[ok[-1]]
        pred = p_test >= cut
        tp = int((pred & (y_test == 1)).sum())
        precision = tp / max(int(pred.sum()), 1)
        rows.append({"target recall": t, "threshold": round(float(cut), 3),
                     "recall on test": round(tp / int(y_test.sum()), 3),
                     "precision on test": round(precision, 3),
                     "% of people flagged": round(100 * pred.mean(), 1),
                     "people flagged per case found": round(1 / precision, 1)})
    return pd.DataFrame(rows).set_index("target recall")


def subgroup_table(y, p, threshold, groups: dict) -> pd.DataFrame:
    """Recall / precision / AUC inside each subgroup, at the single frozen threshold."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for name, mask in groups.items():
        m = np.asarray(mask)
        yy, pp = y[m], p[m]
        pred = pp >= threshold
        tp = int((pred & (yy == 1)).sum())
        rows.append({"group": name, "n": int(m.sum()), "disease rate": round(yy.mean(), 3),
                     "% flagged": round(100 * pred.mean(), 1),
                     "recall": round(tp / max(int(yy.sum()), 1), 3),
                     "precision": round(tp / max(int(pred.sum()), 1), 3),
                     "ROC-AUC": round(roc_auc_score(yy, pp), 3) if 0 < yy.sum() < len(yy) else np.nan})
    return pd.DataFrame(rows).set_index("group")


# ---------------------------------------------------------------- learning curve by training size
def size_learning_curve(make_model, data: dict, fractions=(0.05, 0.10, 0.25, 0.50, 0.75, 1.0),
                        use_engineered: bool = True, scale: bool = True) -> pd.DataFrame:
    """Fit on growing slices of the training set and score on validation.

    Shows whether more data would help: a training/validation gap that closes means
    more data cannot help; one that stays open means it could.
    """
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from src.experiment import build_preprocessor, feature_columns, positive_proba

    cols = feature_columns(use_engineered)
    rows = []
    for frac in fractions:
        if frac < 1.0:
            X_fit, _, y_fit, _ = train_test_split(data["X_train"][cols], data["y_train"], train_size=frac,
                                                  stratify=data["y_train"], random_state=SEED)
        else:
            X_fit, y_fit = data["X_train"][cols], data["y_train"]
        pipe = Pipeline([("prep", build_preprocessor(use_engineered, scale)), ("model", make_model())])
        t0 = time.time()
        pipe.fit(X_fit, y_fit)
        secs = time.time() - t0
        sample = X_fit.sample(n=min(50_000, len(X_fit)), random_state=SEED)
        tr_p = positive_proba(pipe, sample)
        va_p = positive_proba(pipe, data["X_val"][cols])
        rows.append({"fraction": frac, "n_train": len(X_fit),
                     "train_roc_auc": roc_auc_score(y_fit.loc[sample.index], tr_p),
                     "val_roc_auc": roc_auc_score(data["y_val"], va_p),
                     "val_pr_auc": average_precision_score(data["y_val"], va_p),
                     "fit_seconds": round(secs, 1)})
    return pd.DataFrame(rows).set_index("fraction")


def plot_size_curves(curves: dict, path=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    for (name, df), c in zip(curves.items(), COLORS):
        a.plot(df["n_train"], df["train_roc_auc"], ls="--", marker="o", color=c, label=f"{name}: train")
        a.plot(df["n_train"], df["val_roc_auc"], marker="o", color=c, label=f"{name}: validation")
        b.plot(df["n_train"], df["val_pr_auc"], marker="o", color=c, label=name)
    ticks = sorted({int(n) for df in curves.values() for n in df["n_train"]})
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("training rows (log scale)")
    a.set(ylabel="ROC-AUC", title="ROC-AUC against training-set size")
    b.set(ylabel="PR-AUC (validation)", title="PR-AUC against training-set size")
    a.legend(frameon=False, fontsize=8)
    b.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


# ---------------------------------------------------------------- permutation importance
def permutation_table(pipe, X, y, cols, n_repeats: int = 5) -> pd.DataFrame:
    """Drop in AUC when one raw survey column is shuffled (whole pipeline, so encoders are included)."""
    from sklearn.inspection import permutation_importance
    res = permutation_importance(pipe, X[cols], y, scoring={"roc": "roc_auc", "pr": "average_precision"},
                                 n_repeats=n_repeats, random_state=SEED, n_jobs=1)
    out = pd.DataFrame({
        "roc_auc_drop": res["roc"].importances_mean, "roc_auc_drop_sd": res["roc"].importances_std,
        "pr_auc_drop": res["pr"].importances_mean,
    }, index=cols)
    return out.sort_values("roc_auc_drop", ascending=False)


def plot_importance(table: pd.DataFrame, top: int = 15, path=None):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.33 * top + 1.2))
    ax.barh(t.index, t["roc_auc_drop"], xerr=t["roc_auc_drop_sd"], color="#4c72b0", ecolor="#444444")
    ax.set(xlabel="drop in validation ROC-AUC when the column is shuffled",
           title=f"Permutation importance, top {top}")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()


def logistic_odds_table(pipe, top: int = 12, exclude_flags: bool = True) -> pd.DataFrame:
    """Largest coefficients of a fitted logistic-regression pipeline, as odds ratios.

    Missing-value flag columns are left out by default: they come in perfectly opposite
    pairs after one-hot encoding and mostly describe very small groups.
    """
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    t = pd.DataFrame({"input": names, "coefficient": coef, "odds ratio": np.exp(coef)})
    t["input"] = t["input"].str.replace(r"^(num|bin|nom)__", "", regex=True)
    if exclude_flags:
        t = t[~t["input"].str.contains("missingindicator")]
    return t.reindex(t["coefficient"].abs().sort_values(ascending=False).index).head(top).round(3).reset_index(drop=True)
