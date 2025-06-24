import os
import argparse
import sys
from tqdm import tqdm

def delete_files_by_interval(data_dir, interval=3, dry_run=False, extension=".pt"):
    """
    Deletes files in a directory at a regular interval.

    Args:
        data_dir (str): The path to the directory containing the files.
        interval (int): The interval at which to delete files (e.g., 3 means delete every 3rd file).
        dry_run (bool): If True, only print which files would be deleted without actually deleting them.
        extension (str): The file extension to target (e.g., ".pt").
    """
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at {data_dir}")
        return

    print(f"Scanning for files with extension '{extension}' in '{data_dir}'...")
    try:
        # Sort files to ensure consistent deletion order
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(extension)])
    except OSError as e:
        print(f"Error reading directory: {e}")
        return
        
    if not all_files:
        print("No matching files found to delete.")
        return

    # Select every Nth file to be deleted (e.g., if interval is 3, select 3rd, 6th, 9th...)
    files_to_delete = all_files[interval-1::interval]
    
    num_to_delete = len(files_to_delete)
    total_files = len(all_files)

    print(f"Found {total_files} files with extension '{extension}'.")
    print(f"This script will delete {num_to_delete} files (every {interval}rd file).")

    if dry_run:
        print("\n--- DRY RUN ---")
        print("The following files would be deleted (showing first 10 as a sample):")
        for i, f in enumerate(files_to_delete):
            if i < 10:
                print(os.path.join(data_dir, f))
        if num_to_delete > 10:
            print(f"... and {num_to_delete - 10} more.")
        print("\nNo files were actually deleted. Run without the --dry-run flag to proceed.")
        return

    # Confirmation prompt to prevent accidental deletion
    confirm = input(f"\nAre you sure you want to delete {num_to_delete} files from '{data_dir}'? [y/N]: ")
    if confirm.lower() != 'y':
        print("Deletion cancelled by user.")
        return

    print("Deleting files...")
    deleted_count = 0
    failed_count = 0
    for filename in tqdm(files_to_delete, desc="Deleting files"):
        file_path = os.path.join(data_dir, filename)
        try:
            os.remove(file_path)
            deleted_count += 1
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
            failed_count += 1
    
    print("\n--- Deletion Summary ---")
    print(f"Successfully deleted: {deleted_count} files.")
    if failed_count > 0:
        print(f"Failed to delete: {failed_count} files.")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up data files by deleting every Nth file in a specified directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        help="The directory containing the files to be cleaned up."
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=3, 
        help="Interval for deletion. e.g., 3 means delete every 3rd file. Default is 3."
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".pt",
        help="File extension to target. Default is '.pt'."
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Perform a dry run which lists files to be deleted without actually deleting them."
    )

    args = parser.parse_args()
    delete_files_by_interval(args.data_dir, args.interval, args.dry_run, args.ext)

if __name__ == "__main__":
    main() 