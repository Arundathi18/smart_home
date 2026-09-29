"""Local storage for the collected gesture dataset.

Two files live in the project's data/ directory (created automatically):

- landmarks.csv - one row per recorded sample: which recording session it
  came from, when, its gesture label, and its 63 normalized landmark
  feature values.
- sessions.json - one entry per recording session (each run of
  apps/collect_dataset.py), so training can tell which samples came from
  the same sitting in front of the camera.

Keeping session_id alongside every sample is what lets the training code
later split train/validation/test *by session* instead of by individual
frame - see gesture_smart_home/ml/training.py for why that matters.
"""

import csv
import json
import time
import uuid
from pathlib import Path

from gesture_smart_home.gestures.constants import GESTURES, FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

CSV_HEADER = ["session_id", "timestamp", "gesture", *FEATURE_COLUMNS]


class GestureDataset:
    def __init__(self, data_dir=DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / "landmarks.csv"
        self.sessions_path = self.data_dir / "sessions.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

    # --- Sessions -----------------------------------------------------

    def new_session(self, notes=""):
        """Start a new recording session and return its id."""
        session_id = uuid.uuid4().hex[:8]
        sessions = self._load_sessions()
        sessions.append({
            "session_id": session_id,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": None,
            "notes": notes,
            "sample_count": 0,
        })
        self._save_sessions(sessions)
        return session_id

    def end_session(self, session_id, sample_count=0):
        sessions = self._load_sessions()
        for session in sessions:
            if session["session_id"] == session_id:
                session["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                session["sample_count"] = sample_count
        self._save_sessions(sessions)

    def _load_sessions(self):
        if self.sessions_path.exists():
            with open(self.sessions_path) as f:
                return json.load(f)
        return []

    def _save_sessions(self, sessions):
        with open(self.sessions_path, "w") as f:
            json.dump(sessions, f, indent=2)

    def list_sessions(self):
        """Return session records with sample_count and per-gesture counts
        recomputed live from landmarks.csv (authoritative, in case a
        session was interrupted before end_session() ran)."""
        sessions = self._load_sessions()
        gesture_counts_by_session = {}
        with open(self.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                sid = row["session_id"]
                bucket = gesture_counts_by_session.setdefault(sid, {g: 0 for g in GESTURES})
                if row["gesture"] in bucket:
                    bucket[row["gesture"]] += 1

        for session in sessions:
            counts = gesture_counts_by_session.get(session["session_id"], {g: 0 for g in GESTURES})
            session["sample_count"] = sum(counts.values())
            session["gesture_counts"] = counts
        return sessions

    # --- Samples --------------------------------------------------------

    def append_sample(self, session_id, gesture, feature_vector):
        if gesture not in GESTURES:
            raise ValueError(f"Unknown gesture: {gesture!r}")
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([session_id, time.time(), gesture, *feature_vector])

    def counts_by_gesture(self):
        counts = {g: 0 for g in GESTURES}
        with open(self.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["gesture"] in counts:
                    counts[row["gesture"]] += 1
        return counts

    def total_count(self):
        return sum(self.counts_by_gesture().values())

    # --- Delete / reset ---------------------------------------------------

    def reset(self):
        """Delete every collected sample and every session record."""
        if self.csv_path.exists():
            self.csv_path.unlink()
        if self.sessions_path.exists():
            self.sessions_path.unlink()
        self._ensure_csv_header()

    def delete_gesture(self, gesture):
        """Delete all samples for one gesture class, keep everything else."""
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            kept_rows = [row for row in reader if row["gesture"] != gesture]

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

    def delete_session(self, session_id):
        """Delete every sample from one recording session, and remove it
        from sessions.json. Everything from other sessions is untouched."""
        if self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                kept_rows = [row for row in reader if row["session_id"] != session_id]

            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)

        sessions = [s for s in self._load_sessions() if s["session_id"] != session_id]
        self._save_sessions(sessions)
