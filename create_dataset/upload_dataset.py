#!/usr/bin/env python3
"""
Script to upload the eagle-mix dataset to Hugging Face Hub
"""

from datasets import load_from_disk, DatasetDict
from huggingface_hub import HfApi
import os


def upload_dataset():
    # Path to your dataset
    dataset_path = "/nobackup/qinghao/dataset/eagle-mix"

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print("Loading dataset from disk...")
    try:
        # Load the dataset
        dataset = load_from_disk(dataset_path)
        print(f"Dataset loaded successfully")
        print(f"Dataset info: {dataset}")

        # Replace with your desired repository name
        repo_id = "your-username/eagle-mix"  # Change this to your HF username

        print(f"Uploading dataset to {repo_id}...")

        # Upload to Hugging Face Hub
        dataset.push_to_hub(
            repo_id=repo_id,
            private=False,  # Set to True if you want a private dataset
            commit_message="Upload eagle-mix dataset",
        )

        print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"Error loading or uploading dataset: {e}")
        print("Make sure the dataset is in the correct format and you're logged in to HF")


if __name__ == "__main__":
    upload_dataset()
