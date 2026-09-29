"""Saves and loads the trained gesture model, its metadata, and its evaluation.

The saved pipeline (models/gesture_classifier.joblib) bundles the fitted
StandardScaler together with the RandomForestClassifier, so loading it and
calling .predict() automatically applies the exact same preprocessing used
during training - there's no separate "preprocessing config" to keep in
sync by hand.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to a file, don't try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def save_model(pipeline, metadata, model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "gesture_classifier.joblib")
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_model(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    model_path = model_dir / "gesture_classifier.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run "
            "'python -m apps.train_model' first."
        )

    pipeline = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return pipeline, metadata


def save_evaluation(evaluation, model_dir=DEFAULT_MODEL_DIR):
    """Save evaluation.json, a human-readable summary .txt, and a confusion
    matrix image, all under model_dir - handy for pasting into a report."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    _save_text_summary(evaluation, model_dir)
    _save_confusion_matrix_plot(evaluation["test"], model_dir)


def _save_text_summary(evaluation, model_dir):
    lines = []
    for split_name in ("validation", "test"):
        report = evaluation[split_name]
        lines.append(f"=== {split_name.upper()} SET ({report['n_samples']} samples) ===")
        lines.append(f"Accuracy:           {report['accuracy']:.3f}")
        lines.append(f"Precision (macro):  {report['precision_macro']:.3f}")
        lines.append(f"Recall    (macro):  {report['recall_macro']:.3f}")
        lines.append(f"F1        (macro):  {report['f1_macro']:.3f}")
        lines.append("")
        lines.append(report["classification_report"])
        lines.append("")

    with open(model_dir / "evaluation_summary.txt", "w") as f:
        f.write("\n".join(lines))


def _save_confusion_matrix_plot(test_report, model_dir):
    labels = test_report["labels"]
    cm = np.array(test_report["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    ax.set_title(f"Confusion Matrix - test set (n={test_report['n_samples']})")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(model_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
