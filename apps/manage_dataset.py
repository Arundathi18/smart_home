"""Command-line tool for inspecting and pruning recording sessions.

Unlike apps/collect_dataset.py, this has no camera or GUI window - it just
lists what's in data/landmarks.csv and data/sessions.json, and lets you
delete a specific bad session (e.g. one you recorded before realizing the
window didn't have keyboard focus, so it's all one gesture by mistake).

Run from the project root (with the virtual environment activated):

    python -m apps.manage_dataset list
    python -m apps.manage_dataset delete <session_id>
"""

import argparse

from gesture_smart_home.gestures.dataset import GestureDataset


def list_sessions(dataset):
    sessions = dataset.list_sessions()
    if not sessions:
        print("No sessions recorded yet. Run 'python -m apps.collect_dataset' first.")
        return

    print(f"{'SESSION ID':<10} {'STARTED':<20} {'TOTAL':<7} PER-GESTURE COUNTS")
    for s in sessions:
        counts_str = ", ".join(f"{g}={c}" for g, c in s["gesture_counts"].items())
        print(f"{s['session_id']:<10} {s['started_at']:<20} {s['sample_count']:<7} {counts_str}")

    print(f"\nTotal samples across all sessions: {dataset.total_count()}")
    print(f"Overall per-gesture counts: {dataset.counts_by_gesture()}")


def delete_session(dataset, session_id):
    sessions = {s["session_id"]: s for s in dataset.list_sessions()}
    if session_id not in sessions:
        print(f"No session found with id '{session_id}'. Run 'python -m apps.manage_dataset list' to see valid ids.")
        return

    session = sessions[session_id]
    print(f"About to delete session '{session_id}' ({session['sample_count']} samples: {session['gesture_counts']}).")
    confirm = input("This cannot be undone. Type 'y' to confirm: ").strip().lower()
    if confirm == "y":
        dataset.delete_session(session_id)
        print(f"Deleted session '{session_id}'.")
    else:
        print("Cancelled - nothing was deleted.")


def main():
    parser = argparse.ArgumentParser(description="List or delete gesture recording sessions.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List all recording sessions and their sample counts")
    delete_parser = subparsers.add_parser("delete", help="Delete one recording session")
    delete_parser.add_argument("session_id", help="The session id to delete (see 'list')")

    args = parser.parse_args()
    dataset = GestureDataset()

    if args.command == "list":
        list_sessions(dataset)
    elif args.command == "delete":
        delete_session(dataset, args.session_id)


if __name__ == "__main__":
    main()
