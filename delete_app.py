# delete_app.py
"""
Delete all pipeline outputs for one or more apps and remove them from groups.json.

Usage:
    python delete_app.py --app_id com.duolingo
    python delete_app.py --app_id com.duolingo --group_only
    python delete_app.py --list
"""

import argparse
import json
import os
import glob

OUTPUT_DIR = "data/outputs"
GROUPS_FILE = f"{OUTPUT_DIR}/groups.json"


def load_groups() -> list[dict]:
    if not os.path.exists(GROUPS_FILE):
        return []
    with open(GROUPS_FILE) as f:
        return json.load(f)


def save_groups(groups: list[dict]):
    with open(GROUPS_FILE, "w") as f:
        json.dump(groups, f, indent=2)


def list_apps():
    """Print all apps that have output files."""
    files = glob.glob(f"{OUTPUT_DIR}/*_themes.parquet")
    if not files:
        print("No apps found in data/outputs/")
        return

    groups = load_groups()
    main_ids = {g["main_app_id"] for g in groups}

    print(f"\n{'App ID':<45} {'Main app':<10} {'Files'}")
    print("-" * 70)
    for f in sorted(files):
        app_id = (
            os.path.basename(f)
            .replace("_themes.parquet", "")
            .replace("_", ".")
        )
        # Find all output files for this app
        sid = app_id.replace(".", "_")
        app_files = glob.glob(f"{OUTPUT_DIR}/{sid}*")
        is_main = "✅" if app_id in main_ids else "—"
        print(f"{app_id:<45} {is_main:<10} {len(app_files)} files")


def delete_app_files(app_id: str):
    """Delete all parquet and model files for an app."""
    sid = app_id.replace(".", "_")
    files = glob.glob(f"{OUTPUT_DIR}/{sid}*")

    if not files:
        print(f"No files found for {app_id}")
        return 0

    for f in files:
        if os.path.isfile(f):
            os.remove(f)
            print(f"  Deleted: {os.path.basename(f)}")
        elif os.path.isdir(f):
            import shutil
            shutil.rmtree(f)
            print(f"  Deleted directory: {os.path.basename(f)}")

    return len(files)


def remove_from_groups(app_id: str):
    """
    Remove an app from groups.json.
    If it's a main app, removes the whole group.
    If it's a competitor, removes it from the group's app_ids list.
    """
    groups = load_groups()
    updated_groups = []
    removed_from = []

    for g in groups:
        if g["main_app_id"] == app_id:
            # Main app — remove the whole group
            removed_from.append(f"removed group '{g.get('group_name', g['main_app_id'])}'")
        elif app_id in g["app_ids"]:
            # Competitor — remove from app_ids only
            g["app_ids"] = [aid for aid in g["app_ids"] if aid != app_id]
            updated_groups.append(g)
            removed_from.append(
                f"removed from group '{g.get('group_name', g['main_app_id'])}'"
            )
        else:
            updated_groups.append(g)

    if removed_from:
        save_groups(updated_groups)
        for msg in removed_from:
            print(f"  groups.json: {msg}")
    else:
        print(f"  {app_id} not found in groups.json")

    return updated_groups


def delete_app(app_id: str, group_only: bool = False):
    """Full delete — files and group entry."""
    print(f"\nDeleting: {app_id}")

    if not group_only:
        n = delete_app_files(app_id)
        print(f"  {n} file(s) deleted")
    else:
        print("  Skipping file deletion (--group_only)")

    remove_from_groups(app_id)
    print("Done.\n")


def main():
    parser = argparse.ArgumentParser(description="Delete AcquireIQ app outputs")
    parser.add_argument("--app_id", help="App ID to delete e.g. com.duolingo")
    parser.add_argument(
        "--group_only",
        action="store_true",
        help="Only remove from groups.json, keep parquet files"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all apps with output files"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete ALL app outputs and reset groups.json — use with caution"
    )
    args = parser.parse_args()

    if args.list:
        list_apps()
        return

    if args.all:
        confirm = input("This will delete ALL app outputs. Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
        files = glob.glob(f"{OUTPUT_DIR}/*.parquet")
        dirs = glob.glob(f"{OUTPUT_DIR}/*_topic_model")
        for f in files:
            os.remove(f)
        for d in dirs:
            import shutil
            shutil.rmtree(d)
        if os.path.exists(GROUPS_FILE):
            os.remove(GROUPS_FILE)
        print(f"Deleted {len(files)} files and {len(dirs)} model directories.")
        return

    if not args.app_id:
        parser.print_help()
        return

    # Confirm before deleting
    if not args.group_only:
        sid = args.app_id.replace(".", "_")
        files = glob.glob(f"{OUTPUT_DIR}/{sid}*")
        if files:
            print(f"\nFiles to be deleted for {args.app_id}:")
            for f in files:
                size = (
                    os.path.getsize(f) // 1024
                    if os.path.isfile(f)
                    else sum(
                        os.path.getsize(os.path.join(dp, fn))
                        for dp, _, filenames in os.walk(f)
                        for fn in filenames
                    ) // 1024
                )
                print(f"  {os.path.basename(f)} ({size}KB)")

        confirm = input(f"\nDelete all outputs for {args.app_id}? (y/n): ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    delete_app(args.app_id, group_only=args.group_only)


if __name__ == "__main__":
    main()