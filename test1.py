import datasets
import numpy as np
from collections import Counter


def analyze_length_distribution(dataset, dataset_name, text_field=None):
    """Analyze length distribution of a dataset"""
    print(f"\n=== {dataset_name} Length Distribution ===")

    lengths = []

    # Extract text lengths based on dataset structure
    for item in dataset:
        if text_field:
            # For datasets with a specific text field
            if isinstance(item[text_field], list):
                # If it's a list of messages/conversations
                total_length = sum(len(str(msg)) for msg in item[text_field])
            else:
                total_length = len(str(item[text_field]))
        else:
            # Try to infer the text content
            if "conversations" in item:
                total_length = sum(len(str(msg.get("value", ""))) for msg in item["conversations"])
            elif "messages" in item:
                total_length = sum(len(str(msg.get("content", ""))) for msg in item["messages"])
            elif "text" in item:
                total_length = len(str(item["text"]))
            else:
                # Fallback: sum all string values
                total_length = sum(len(str(v)) for v in item.values() if isinstance(v, str))

        lengths.append(total_length)

    lengths = np.array(lengths)

    # Print statistics
    print(f"Total samples: {len(lengths)}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    print(f"Std deviation: {np.std(lengths):.2f}")

    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(lengths, p):.0f}")

    return lengths


# Load ShareGPT dataset
print("Loading ShareGPT dataset...")
sharegpt_dataset = datasets.load_dataset(
    "json", data_files="/nobackup/qinghao/trace/ShareGPT_V4.3_unfiltered_cleaned_split.json", split="train"
)
print(f"ShareGPT dataset size: {len(sharegpt_dataset)}")

# Load UltraChat dataset
print("Loading UltraChat dataset...")
ultrachat_dataset = datasets.load_dataset("parquet", data_dir="/nobackup/qinghao/dataset/ultrachat_200k", split="train_sft")
print(f"UltraChat dataset size: {len(ultrachat_dataset)}")

# Load OpenThoughts2-1M dataset
print("Loading OpenThoughts2-1M dataset...")
openthoughts_dataset = datasets.load_dataset(
    "parquet", data_dir="/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M", split="train"
)
print(f"OpenThoughts2-1M dataset size: {len(openthoughts_dataset)}")

# Analyze length distributions
print("\n" + "=" * 50)
print("ANALYZING LENGTH DISTRIBUTIONS")
print("=" * 50)

# Analyze each dataset
sharegpt_lengths = analyze_length_distribution(sharegpt_dataset, "ShareGPT")
ultrachat_lengths = analyze_length_distribution(ultrachat_dataset, "UltraChat")
openthoughts_lengths = analyze_length_distribution(openthoughts_dataset, "OpenThoughts2-1M")

# Summary comparison
print("\n" + "=" * 50)
print("SUMMARY COMPARISON")
print("=" * 50)
print(f"{'Dataset':<15} {'Count':<10} {'Mean':<10} {'Median':<10} {'Max':<10}")
print("-" * 55)
print(
    f"{'ShareGPT':<15} {len(sharegpt_lengths):<10} {np.mean(sharegpt_lengths):<10.0f} {np.median(sharegpt_lengths):<10.0f} {np.max(sharegpt_lengths):<10.0f}"
)
print(
    f"{'UltraChat':<15} {len(ultrachat_lengths):<10} {np.mean(ultrachat_lengths):<10.0f} {np.median(ultrachat_lengths):<10.0f} {np.max(ultrachat_lengths):<10.0f}"
)
print(
    f"{'OpenThoughts2-1M':<15} {len(openthoughts_lengths):<10} {np.mean(openthoughts_lengths):<10.0f} {np.median(openthoughts_lengths):<10.0f} {np.max(openthoughts_lengths):<10.0f}"
)
