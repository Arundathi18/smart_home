"""Trains a Random Forest gesture classifier on the collected landmark dataset.

Why split by *session* instead of randomly by row: consecutive frames from
the same recording session look almost identical (your hand barely moves
between two frames captured 1/30th of a second apart). A plain random
train/test split would scatter near-duplicate frames across both sides of
the split, so the model could "recognize" a training frame's near-twin in
the test set rather than genuinely generalizing - producing a test accuracy
that looks great but doesn't mean much. Splitting by session_id instead
guarantees every sample from a given recording session lands entirely in
one of train, validation, or test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from gesture_smart_home import config
from gesture_smart_home.gestures.constants import GESTURES, FEATURE_COLUMNS
from gesture_smart_home.gestures.dataset import GestureDataset
from gesture_smart_home.ml import model_io


class InsufficientDataError(RuntimeError):
    """Raised when the dataset is too small or too repetitive to train on
    and get a test accuracy that can actually be trusted."""


def load_dataframe(csv_path=None):
    csv_path = Path(csv_path) if csv_path else GestureDataset().csv_path
    if not csv_path.exists():
        raise InsufficientDataError(
            f"No dataset found at {csv_path}. Run 'python -m apps.collect_dataset' "
            "to collect samples first."
        )
    return pd.read_csv(csv_path).reset_index(drop=True)


def _sessions_with_full_gesture_coverage(df):
    """Session ids whose recorded rows include every gesture in GESTURES.

    A session recorded with only one or two gestures selected (e.g. a
    targeted top-up session) is fine as *training* data, but it can't fairly
    serve as validation or test data - a test set missing a whole gesture
    can't tell you anything about how well the model recognizes that
    gesture. Restricting val/test candidates to "complete" sessions avoids
    silently producing a held-out set that's blind to some classes.
    """
    all_gestures = frozenset(GESTURES)
    gesture_sets = df.groupby("session_id")["gesture"].agg(lambda s: frozenset(s.unique()))
    return set(gesture_sets[gesture_sets == all_gestures].index)


def _check_data_is_sufficient(df):
    problems = []

    n_sessions = df["session_id"].nunique()
    if n_sessions < config.MIN_SESSIONS_REQUIRED:
        problems.append(
            f"Only {n_sessions} recording session(s) found, need at least "
            f"{config.MIN_SESSIONS_REQUIRED}. Run 'python -m apps.collect_dataset' "
            "again on a different occasion (slightly different hand position, "
            "distance from the camera, or lighting works well) - each run "
            "creates a new session, and a genuinely held-out test set needs "
            "sessions the model has never seen."
        )

    counts = df["gesture"].value_counts()
    for gesture in GESTURES:
        n = int(counts.get(gesture, 0))
        if n < config.MIN_SAMPLES_PER_GESTURE:
            problems.append(
                f"Only {n} samples for '{gesture}', need at least "
                f"{config.MIN_SAMPLES_PER_GESTURE}. Collect more with "
                "'python -m apps.collect_dataset'."
            )

    complete_sessions = _sessions_with_full_gesture_coverage(df)
    if len(complete_sessions) < 2:
        problems.append(
            f"Only {len(complete_sessions)} recording session(s) contain samples "
            "for all 4 gestures, need at least 2 - validation and test sets are "
            "only built from sessions that cover every gesture, so each can be "
            "fairly evaluated on all 4. Sessions recording just one or two "
            "gestures (e.g. a targeted top-up session) still help training, "
            "just not validation/test. Record a session with all 4 gestures "
            "using 'python -m apps.collect_dataset'."
        )

    if problems:
        raise InsufficientDataError("Not enough data to train reliably:\n- " + "\n- ".join(problems))


def _group_aware_split(df, test_size, val_size, random_state):
    """Return (train_positions, val_positions, test_positions) as integer
    arrays into df, with no session_id shared across the three groups, and
    with validation/test built only from sessions that cover all 4 gestures
    (see _sessions_with_full_gesture_coverage)."""
    groups = df["session_id"].values
    complete_sessions = _sessions_with_full_gesture_coverage(df)

    complete_mask = np.isin(groups, list(complete_sessions))
    complete_pos = np.where(complete_mask)[0]
    incomplete_pos = np.where(~complete_mask)[0]

    complete_groups = groups[complete_pos]
    dummy_X = np.zeros(len(complete_pos))

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_rel, test_rel = next(test_splitter.split(dummy_X, groups=complete_groups))
    test_pos = complete_pos[test_rel]
    train_val_pos = complete_pos[train_val_rel]

    train_val_groups = groups[train_val_pos]
    if len(set(train_val_groups)) < 2:
        # Not enough distinct complete sessions remain to also carve out a
        # separate validation set. Fall back to validating on the training
        # data (the held-out test set from above is unaffected and still
        # valid). Incomplete sessions still go to train either way.
        train_pos = np.concatenate([train_val_pos, incomplete_pos])
        return train_pos, train_val_pos, test_pos

    relative_val_size = val_size / (1 - test_size)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_pos_rel, val_pos_rel = next(val_splitter.split(np.zeros(len(train_val_pos)), groups=train_val_groups))

    train_pos = np.concatenate([train_val_pos[train_pos_rel], incomplete_pos])
    val_pos = train_val_pos[val_pos_rel]
    return train_pos, val_pos, test_pos


def _assert_no_session_leakage(df, train_pos, val_pos, test_pos):
    session_values = df["session_id"].values
    train_sessions = set(session_values[train_pos])
    val_sessions = set(session_values[val_pos])
    test_sessions = set(session_values[test_pos])
    assert not (train_sessions & test_sessions), "Session leaked between train and test!"
    assert not (val_sessions & test_sessions), "Session leaked between val and test!"


def _evaluate(pipeline, X, y, split_name):
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=GESTURES, average="macro", zero_division=0
    )
    return {
        "split": split_name,
        "n_samples": int(len(y)),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": classification_report(y, y_pred, labels=GESTURES, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=GESTURES).tolist(),
        "labels": GESTURES,
    }


def train_and_save(csv_path=None, model_dir=model_io.DEFAULT_MODEL_DIR):
    df = load_dataframe(csv_path)
    _check_data_is_sufficient(df)

    train_pos, val_pos, test_pos = _group_aware_split(
        df,
        test_size=config.TEST_SPLIT_SIZE,
        val_size=config.VAL_SPLIT_SIZE,
        random_state=config.RANDOM_STATE,
    )
    _assert_no_session_leakage(df, train_pos, val_pos, test_pos)

    X = df[FEATURE_COLUMNS].values
    y = df["gesture"].values

    X_train, y_train = X[train_pos], y[train_pos]
    X_val, y_val = X[val_pos], y[val_pos]
    X_test, y_test = X[test_pos], y[test_pos]

    # StandardScaler isn't strictly necessary for a Random Forest (it's
    # scale-invariant by nature) - it's included so the pipeline stays
    # correct if the classifier is swapped for something scale-sensitive
    # later. The real fix for camera-distance/position sensitivity is the
    # wrist-relative, scale-normalized landmarks already computed in
    # gesture_smart_home/gestures/features.py before they ever reach here.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X_train, y_train)

    evaluation = {
        "validation": _evaluate(pipeline, X_val, y_val, "validation"),
        "test": _evaluate(pipeline, X_test, y_test, "test"),
    }

    metadata = {
        "gestures": GESTURES,
        "feature_columns": FEATURE_COLUMNS,
        "n_train": int(len(train_pos)),
        "n_val": int(len(val_pos)),
        "n_test": int(len(test_pos)),
        "n_sessions": int(df["session_id"].nunique()),
        "random_state": config.RANDOM_STATE,
    }

    model_io.save_model(pipeline, metadata, model_dir=model_dir)
    model_io.save_evaluation(evaluation, model_dir=model_dir)

    return pipeline, metadata, evaluation
