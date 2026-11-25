"""Evaluation of the trained pneumonia models on the held-out test set."""

import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src import config
from src.data_loader import load_split, validate_dataset
from src.predict import model_exists


def _prepare_test_data(model_name: str):
    x_test, y_test, _ = load_split("test", model_name=model_name)
    return x_test, y_test


def evaluate_model(model, x_test, y_test, model_name: str) -> dict:
    """Compute all clinical/classification metrics for a model on test data."""
    y_true = np.asarray(y_test, dtype=int).ravel()
    probs = model.predict(np.asarray(x_test, dtype=np.float32), verbose=0)
    prob = np.asarray(probs).ravel().astype(float)
    y_pred = (prob >= 0.5).astype(int)

    result = {
        "model": model_name,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1": None,
        "roc_auc": None,
        "confusion_matrix": None,
        "classification_report": None,
        "roc": None,
        "predictions": [int(v) for v in y_pred],
        "probabilities": [float(v) for v in prob],
    }

    try:
        result["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        result["accuracy"] = None
    try:
        result["precision"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    except Exception:
        result["precision"] = None
    try:
        result["recall"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    except Exception:
        result["recall"] = None
    try:
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        denom = tn + fp
        result["specificity"] = float(tn / denom) if denom > 0 else 0.0
    except Exception:
        result["specificity"] = None
    try:
        result["f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    except Exception:
        result["f1"] = None

    try:
        fpr, tpr, thresholds = roc_curve(y_true, prob)
        result["roc_auc"] = float(auc(fpr, tpr))
        result["roc"] = {"fpr": [float(v) for v in fpr], "tpr": [float(v) for v in tpr], "thresholds": [float(v) for v in thresholds]}
    except Exception:
        result["roc_auc"] = None
        result["roc"] = None

    try:
        result["confusion_matrix"] = [[int(v) for v in row] for row in confusion_matrix(y_true, y_pred, labels=[0, 1])]
    except Exception:
        result["confusion_matrix"] = None

    try:
        result["classification_report"] = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)
    except Exception:
        result["classification_report"] = None

    return result


def save_evaluation_plots(metrics_by_model: dict, path_prefix: str = "evaluation") -> dict:
    """Save confusion matrix and ROC plots for each evaluated model."""
    plots_dir = config.PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, metrics in metrics_by_model.items():
        cm = metrics.get("confusion_matrix")
        roc = metrics.get("roc") or {}
        fpr = roc.get("fpr") or []
        tpr = roc.get("tpr") or []
        if cm is not None and len(fpr) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(cm, cmap="Blues")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("Actual")
            axes[0].set_title(f"{name} - Confusion Matrix")
            axes[0].set_xticks([0, 1])
            axes[0].set_yticks([0, 1])
            axes[0].set_xticklabels(["NORMAL", "PNEUMONIA"])
            axes[0].set_yticklabels(["NORMAL", "PNEUMONIA"])
            plt.colorbar(axes[0], ax=axes[0])
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUC = {metrics.get('roc_auc')}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{name} - ROC Curve")
            ax.legend()
            plt.tight_layout()
            out = plots_dir / f"{path_prefix}_{name}.png"
            fig.savefig(out)
            plt.close(fig)
            saved[name] = str(out)
    return saved


def load_evaluation_metrics() -> dict:
    """Load the saved evaluation metrics (if available)."""
    path = config.EVALUATION_METRICS_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def run() -> int:
    """Evaluate every trained model on the held-out test set."""
    available = [m for m in ("cnn", "mobilenetv2") if model_exists(m)]
    if not available:
        print("No trained models found. Please train a model first using:")
        print("  python -m src.train --model cnn")
        print("  python -m src.train --model mobilenetv2")
        return 1

    metrics_by_model = {}
    validation_metrics_by_model = {}
    for name in available:
        from src.predict import load_model

        model = load_model(name)
        x_test, y_test = _prepare_test_data(name)
        metrics = evaluate_model(model, x_test, y_test, name)
        metrics_by_model[name] = metrics
        print(f"Test metrics for {name}: {metrics}")

    plots_saved = save_evaluation_plots(metrics_by_model)

    # Model selection based on validation performance (not test performance).
    validation_metrics_by_model = {}
    best_model = None
    best_score = None
    for name in available:
        try:
            from src.predict import load_model

            model = load_model(name)
            x_val, y_val, _ = load_split("val", model_name=name)
            val_metrics = evaluate_model(model, x_val, y_val, name)
            validation_metrics_by_model[name] = val_metrics
            score = val_metrics.get("f1") or 0.0
            if best_score is None or score > best_score:
                best_score = score
                best_model = name
        except Exception as exc:
            print(f"Validation evaluation failed for {name}: {exc}")

    evaluation = {
        "generated_at": datetime.now().isoformat(),
        "dataset": validate_dataset(),
        "models": metrics_by_model,
        "validation_models": validation_metrics_by_model,
        "best_model": best_model,
        "recommended_model": best_model,
        "plots_saved": plots_saved,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EVALUATION_METRICS_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)

    with open(config.RESULTS_DIR / "best_model.json", "w") as f:
        json.dump(
            {
                "best_model": best_model,
                "selected_by": "validation_f1",
                "validation_metrics": validation_metrics_by_model,
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
