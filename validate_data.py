#!/usr/bin/env python3
"""
Script to validate PyTorch data files and identify corrupted ones
"""
import os
import torch
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import logging
import psutil


def validate_single_file(filepath):
    """Validate a single PyTorch file with memory optimization"""
    try:
        # Use memory mapping for large files and set weights_only for security
        data = torch.load(filepath)
        # Explicitly delete to free memory immediately
        # del data
        return filepath, True, None
    except Exception as e:
        return filepath, False, str(e)


def get_optimal_workers():
    """Get optimal number of workers based on system resources"""
    cpu_count = mp.cpu_count()
    # Consider memory constraints - PyTorch files can be large
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Estimate workers based on available memory (assuming ~1GB per worker)
    memory_based_workers = max(1, int(available_memory_gb * 0.8))

    # Use the minimum of CPU count and memory-based estimate
    optimal_workers = min(cpu_count, memory_based_workers)

    print(f"System info: {cpu_count} CPUs, {available_memory_gb:.1f}GB available memory")
    print(f"Recommended workers: {optimal_workers}")

    return optimal_workers


def validate_data_directory(data_path, num_workers=None):
    """
    Validate all .pt files in the data directory

    Args:
        data_path: Path to the data directory
        num_workers: Number of parallel workers for validation (auto-detect if None)
    """
    data_path = Path(data_path)

    # Auto-detect optimal workers if not specified
    if num_workers is None:
        num_workers = get_optimal_workers()

    print(f"Using {num_workers} worker processes")

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

    # Validate files in parallel with chunked processing for better memory management
    chunk_size = max(1, len(pt_files) // (num_workers * 4))  # Process in smaller chunks

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(validate_single_file, pt_files, chunksize=chunk_size), total=len(pt_files), desc="Validating files")
        )

    # Process results
    for filepath, is_valid, error in results:
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
    parser.add_argument("data_path", help="Path to the data directory")
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of workers for parallel validation (auto-detect if not specified)"
    )

    args = parser.parse_args()

    validate_data_directory(args.data_path, args.workers)


if __name__ == "__main__":
    main()


# Examples:
# python validate_data.py /nobackup/qinghao/runs/eagle/eagle-data/Eagle-Mix-Llama-3.1-8B-Instruct
# python validate_data.py /path/to/data --workers 16
# python validate_data.py /path/to/data --workers $(nproc)
