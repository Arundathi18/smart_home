"""Trains the gesture classifier on the collected dataset and saves it.

Run from the project root (with the virtual environment activated):
    python -m apps.train_model

Requires a dataset collected with 'python -m apps.collect_dataset' across at
least 3 separate recording sessions, with at least 30 samples per gesture -
see gesture_smart_home/ml/training.py for why. Prints accuracy, precision,
recall and F1 on a genuinely held-out test set (no recording session
appears in more than one of train/validation/test), and saves:

    models/gesture_classifier.joblib   the trained model (+ preprocessing)
    models/metadata.json               gesture list, feature layout, split sizes
    models/evaluation.json             full metrics + confusion matrix (machine-readable)
    models/evaluation_summary.txt      the same metrics, readable/paste-able
    models/confusion_matrix.png        confusion matrix plot, for your report
"""

from gesture_smart_home.ml.training import train_and_save, InsufficientDataError


def _print_report(name, report):
    print(f"\n--- {name.upper()} SET ({report['n_samples']} samples) ---")
    print(f"Accuracy:           {report['accuracy']:.3f}")
    print(f"Precision (macro):  {report['precision_macro']:.3f}")
    print(f"Recall    (macro):  {report['recall_macro']:.3f}")
    print(f"F1        (macro):  {report['f1_macro']:.3f}")
    print(report["classification_report"])


def main():
    try:
        pipeline, metadata, evaluation = train_and_save()
    except InsufficientDataError as e:
        print("Cannot train yet:\n")
        print(str(e))
        return

    print(
        f"Trained on {metadata['n_train']} samples, "
        f"validated on {metadata['n_val']}, "
        f"tested on {metadata['n_test']} "
        f"(from {metadata['n_sessions']} recording sessions)."
    )

    _print_report("validation", evaluation["validation"])
    _print_report("test", evaluation["test"])

    print("\nSaved to models/: gesture_classifier.joblib, metadata.json,")
    print("evaluation.json, evaluation_summary.txt, confusion_matrix.png")


if __name__ == "__main__":
    main()
