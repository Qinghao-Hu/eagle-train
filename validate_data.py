#!/usr/bin/env python3
"""
Script to validate PyTorch data files and identify corrupted ones
"""
import os
import torch
from pathlib import Path
from tqdm import tqdm
import logging


def validate_single_file(filepath):
    """Validate a single PyTorch file"""
    try:
        # Use memory mapping for large files and set weights_only for security
        data = torch.load(filepath)
        # Explicitly delete to free memory immediately
        # del data
        return filepath, True, None
    except Exception as e:
        return filepath, False, str(e)


def validate_data_directory(data_path):
    """
    Validate all .pt files in the data directory

    Args:
        data_path: Path to the data directory
    """
    data_path = Path(data_path)

    # Find all .pt files
    pt_files = list(data_path.rglob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in {data_path}")
        return

    print(f"Found {len(pt_files)} .pt files to validate")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("data_validation.log"), logging.StreamHandler()],
    )

    corrupted_files = []

    # Validate files sequentially
    for filepath in tqdm(pt_files, desc="Validating files"):
        filepath, is_valid, error = validate_single_file(filepath)

        if not is_valid:
            corrupted_files.append((filepath, error))
            logging.error(f"CORRUPTED: {filepath} - {error}")

    # Summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total files: {len(pt_files)}")
    print(f"Valid files: {len(pt_files) - len(corrupted_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")

    if corrupted_files:
        print(f"\n=== CORRUPTED FILES ===")
        for filepath, error in corrupted_files:
            print(f"❌ {filepath}")
            print(f"   Error: {error}")

        # Save corrupted file list
        with open("corrupted_files.txt", "w") as f:
            for filepath, error in corrupted_files:
                f.write(f"{filepath}\n")

        print(f"\nCorrupted file list saved to: corrupted_files.txt")
        print(f"Consider removing or regenerating these files before resuming training.")
    else:
        print("✅ All files are valid!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate PyTorch data files")
    parser.add_argument(
        "--data_path",
        default="/nobackup/qinghao/runs/eagle/eagle-data/Eagle-Mix-Llama-3.1-8B-Instruct",
        type=str,
        help="Path to the data directory",
    )

    args = parser.parse_args()

    validate_data_directory(args.data_path)


if __name__ == "__main__":
    main()


# Examples:
# python validate_data.py /nobackup/qinghao/runs/eagle/eagle-data/Eagle-Mix-Llama-3.1-8B-Instruct
# python validate_data.py /path/to/data
