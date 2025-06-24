#!/usr/bin/env python3
"""
Script to analyze sequence length statistics for three datasets:
- ShareGPT
- UltraChat
- OpenThoughts2-1M
"""

import datasets
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns
from collections import defaultdict
import json
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import os


def analyze_sharegpt_structure(dataset, num_samples=5):
    """Analyze the structure of ShareGPT dataset"""
    print("\n=== ShareGPT Dataset Structure ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")

    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}:")
        sample = dataset[i]
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value)} - {value}")

    return dataset.column_names


def analyze_ultrachat_structure(dataset, num_samples=5):
    """Analyze the structure of UltraChat dataset"""
    print("\n=== UltraChat Dataset Structure ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")

    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}:")
        sample = dataset[i]
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if value and isinstance(value[0], dict):
                    print(f"    First item keys: {list(value[0].keys())}")
            else:
                print(f"  {key}: {type(value)} - {value}")

    return dataset.column_names


def analyze_openthoughts_structure(dataset, num_samples=5):
    """Analyze the structure of OpenThoughts2-1M dataset"""
    print("\n=== OpenThoughts2-1M Dataset Structure ===")
    print(f"Dataset size: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")

    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}:")
        sample = dataset[i]
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if value and isinstance(value[0], dict):
                    print(f"    First item keys: {list(value[0].keys())}")
            else:
                print(f"  {key}: {type(value)} - {value}")

    return dataset.column_names


def extract_conversation_text(sample, dataset_type):
    """Extract conversation text from a sample based on dataset type"""
    text_parts = []

    if dataset_type == "sharegpt":
        if "conversations" in sample and sample["conversations"]:
            # Join all conversation turns
            for turn in sample["conversations"]:
                if isinstance(turn, dict) and "value" in turn and turn["value"]:
                    text_parts.append(str(turn["value"]))
        elif "text" in sample and sample["text"]:
            return str(sample["text"])

    elif dataset_type == "ultrachat":
        if "messages" in sample and sample["messages"]:
            # Join all message content
            for msg in sample["messages"]:
                if isinstance(msg, dict) and "content" in msg and msg["content"]:
                    text_parts.append(str(msg["content"]))
        elif "text" in sample and sample["text"]:
            return str(sample["text"])

    elif dataset_type == "openthoughts":
        # Try multiple possible field names for OpenThoughts dataset
        possible_fields = ["messages", "conversation", "text", "conversations"]

        for field in possible_fields:
            if field in sample and sample[field]:
                if field == "messages" and isinstance(sample[field], list):
                    # Handle messages format
                    for msg in sample[field]:
                        if isinstance(msg, dict):
                            # Try different content field names
                            for content_field in ["content", "text", "message", "value"]:
                                if content_field in msg and msg[content_field]:
                                    text_parts.append(str(msg[content_field]))
                                    break
                        elif isinstance(msg, str):
                            text_parts.append(str(msg))
                    if text_parts:
                        break
                elif isinstance(sample[field], str):
                    return str(sample[field])
                elif isinstance(sample[field], list) and sample[field] and isinstance(sample[field][0], str):
                    text_parts = [str(x) for x in sample[field] if x]
                    break

    # If we found text parts, join them
    if text_parts:
        return " ".join(text_parts)

    # Fallback - try to find any text field with more comprehensive search
    for key, value in sample.items():
        if value and (
            "text" in key.lower()
            or "content" in key.lower()
            or "message" in key.lower()
            or "conversation" in key.lower()
            or "dialogue" in key.lower()
        ):
            if isinstance(value, str):
                return str(value)
            elif isinstance(value, list):
                if value and isinstance(value[0], str):
                    return " ".join(str(x) for x in value if x)
                elif value and isinstance(value[0], dict):
                    # Try to extract text from list of dictionaries
                    for item in value:
                        if isinstance(item, dict):
                            for content_field in ["content", "text", "message", "value"]:
                                if content_field in item and item[content_field]:
                                    text_parts.append(str(item[content_field]))
                    if text_parts:
                        return " ".join(text_parts)

    return ""


def calculate_sequence_statistics(lengths, dataset_name):
    """Calculate comprehensive statistics for sequence lengths"""
    lengths = np.array(lengths)

    # Handle empty arrays
    if len(lengths) == 0:
        print(f"Warning: No valid samples found for {dataset_name}")
        return {
            "dataset": dataset_name,
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "percentile_25": 0.0,
            "percentile_75": 0.0,
            "percentile_90": 0.0,
            "percentile_95": 0.0,
            "percentile_99": 0.0,
        }

    stats = {
        "dataset": dataset_name,
        "count": len(lengths),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "percentile_25": float(np.percentile(lengths, 25)),
        "percentile_75": float(np.percentile(lengths, 75)),
        "percentile_90": float(np.percentile(lengths, 90)),
        "percentile_95": float(np.percentile(lengths, 95)),
        "percentile_99": float(np.percentile(lengths, 99)),
    }

    return stats


def analyze_dataset_lengths(dataset, dataset_name, dataset_type, tokenizer, max_samples=None):
    """Analyze sequence lengths for a dataset"""
    print(f"\n=== Analyzing {dataset_name} ===")

    char_lengths = []
    token_lengths = []
    word_lengths = []
    empty_samples = 0

    dataset_size = len(dataset)
    if max_samples:
        dataset_size = min(max_samples, dataset_size)
        print(f"Analyzing {dataset_size} samples (limited from {len(dataset)})")
    else:
        print(f"Analyzing all {dataset_size} samples")

    for i in tqdm(range(dataset_size), desc=f"Processing {dataset_name}"):
        sample = dataset[i]
        text = extract_conversation_text(sample, dataset_type)

        if text:
            # Character length
            char_lengths.append(len(text))

            # Word length (simple whitespace split)
            word_lengths.append(len(text.split()))

            # Token length
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_lengths.append(len(tokens))
            except Exception as e:
                # If tokenization fails, skip this sample
                print(f"Tokenization error for sample {i}: {e}")
                token_lengths.append(0)
        else:
            empty_samples += 1
            # Debug: Print first few failed extractions
            if empty_samples <= 5:
                print(f"Debug: Sample {i} - Failed to extract text. Keys: {list(sample.keys())}")
                # Print first few characters of each field to understand the structure
                for key, value in sample.items():
                    if isinstance(value, str):
                        print(f"  {key}: '{value[:50]}{'...' if len(value) > 50 else ''}'")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"  {key}: list[{len(value)}] - first item: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"    First item keys: {list(value[0].keys())}")
                    else:
                        print(f"  {key}: {type(value)}")

    print(f"Valid samples with text: {len(char_lengths)}, Empty samples: {empty_samples}")

    if len(char_lengths) == 0:
        print(
            f"Error: No valid text extracted from {dataset_name}. This suggests a problem with the dataset format or text extraction logic."
        )
        return None

    # Calculate statistics
    char_stats = calculate_sequence_statistics(char_lengths, f"{dataset_name} (characters)")
    word_stats = calculate_sequence_statistics(word_lengths, f"{dataset_name} (words)")
    token_stats = calculate_sequence_statistics(token_lengths, f"{dataset_name} (tokens)")

    return {
        "char_stats": char_stats,
        "word_stats": word_stats,
        "token_stats": token_stats,
        "char_lengths": char_lengths,
        "word_lengths": word_lengths,
        "token_lengths": token_lengths,
    }


def print_statistics(stats):
    """Print statistics in a formatted way"""
    print(f"\n--- {stats['dataset']} ---")
    print(f"Count: {stats['count']:,}")
    print(f"Min: {stats['min']:,}")
    print(f"Max: {stats['max']:,}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Std Dev: {stats['std']:.2f}")
    print(f"25th percentile: {stats['percentile_25']:.2f}")
    print(f"75th percentile: {stats['percentile_75']:.2f}")
    print(f"90th percentile: {stats['percentile_90']:.2f}")
    print(f"95th percentile: {stats['percentile_95']:.2f}")
    print(f"99th percentile: {stats['percentile_99']:.2f}")


def create_visualizations(all_results, output_dir="sequence_analysis_plots"):
    """Create visualizations for sequence length analysis"""
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Distribution plots for each metric type
    metrics = ["char_lengths", "word_lengths", "token_lengths"]
    metric_names = ["Character Lengths", "Word Lengths", "Token Lengths"]

    for metric, metric_name in zip(metrics, metric_names):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{metric_name} Distribution Across Datasets", fontsize=16)

        # Collect data for all datasets
        all_data = {}
        for dataset_name, results in all_results.items():
            all_data[dataset_name] = results[metric]

        # Histogram
        ax = axes[0, 0]
        for dataset_name, lengths in all_data.items():
            ax.hist(lengths, bins=50, alpha=0.7, label=dataset_name, density=True)
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Density")
        ax.set_title("Distribution (Histogram)")
        ax.legend()
        ax.set_xlim(0, np.percentile(list(all_data.values())[0], 95))  # Limit to 95th percentile for better visibility

        # Box plot
        ax = axes[0, 1]
        data_for_box = [lengths for lengths in all_data.values()]
        labels_for_box = list(all_data.keys())
        ax.boxplot(data_for_box, labels=labels_for_box)
        ax.set_ylabel(metric_name)
        ax.set_title("Box Plot")
        ax.tick_params(axis="x", rotation=45)

        # Violin plot
        ax = axes[1, 0]
        data_for_violin = []
        labels_for_violin = []
        for dataset_name, lengths in all_data.items():
            data_for_violin.extend(lengths)
            labels_for_violin.extend([dataset_name] * len(lengths))

        # Use seaborn for violin plot
        import pandas as pd

        violin_df = pd.DataFrame({"lengths": data_for_violin, "dataset": labels_for_violin})
        sns.violinplot(data=violin_df, x="dataset", y="lengths", ax=ax)
        ax.set_ylabel(metric_name)
        ax.set_title("Violin Plot")
        ax.tick_params(axis="x", rotation=45)

        # CDF plot
        ax = axes[1, 1]
        for dataset_name, lengths in all_data.items():
            sorted_lengths = np.sort(lengths)
            y = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
            ax.plot(sorted_lengths, y, label=dataset_name, linewidth=2)
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Cumulative Distribution Function")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 2. Comparison table plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for table
    table_data = []
    headers = ["Dataset", "Type", "Count", "Min", "Max", "Mean", "Median", "95th %ile"]

    for dataset_name, results in all_results.items():
        for stats_type in ["char_stats", "word_stats", "token_stats"]:
            stats = results[stats_type]
            type_name = stats_type.replace("_stats", "").capitalize()
            row = [
                dataset_name,
                type_name,
                f"{stats['count']:,}",
                f"{stats['min']:,}",
                f"{stats['max']:,}",
                f"{stats['mean']:.0f}",
                f"{stats['median']:.0f}",
                f"{stats['percentile_95']:.0f}",
            ]
            table_data.append(row)

    # Create table
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color rows alternately
    for i in range(len(table_data)):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i + 1, j)].set_facecolor("#f0f0f0")

    plt.title("Sequence Length Statistics Summary", fontsize=16, pad=20)
    plt.savefig(f"{output_dir}/summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nVisualization plots saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze sequence lengths of three datasets")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum number of samples to analyze per dataset (default: all)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="/nobackup/model/llama3.1/Llama-3.1-8B-Instruct",
        help="Tokenizer to use for token-level analysis",
    )
    parser.add_argument("--output-dir", type=str, default="sequence_analysis_results", help="Directory to save results")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print("Loading datasets...")

    try:
        sharegpt_dataset = None
        # sharegpt_dataset = datasets.load_dataset(
        #     "json", data_files="/nobackup/qinghao/trace/ShareGPT_V4.3_unfiltered_cleaned_split.json", split="train"
        # )
        # print(f"ShareGPT dataset size: {len(sharegpt_dataset)}")
    except Exception as e:
        print(f"Error loading ShareGPT dataset: {e}")
        sharegpt_dataset = None

    try:
        ultrachat_dataset = None
        # ultrachat_dataset = datasets.load_dataset(
        #     "parquet", data_dir="/nobackup/qinghao/dataset/ultrachat_200k", split="train_sft"
        # )
        # print(f"UltraChat dataset size: {len(ultrachat_dataset)}")
    except Exception as e:
        print(f"Error loading UltraChat dataset: {e}")
        ultrachat_dataset = None

    try:
        openthoughts_dataset = datasets.load_dataset(
            "parquet", data_dir="/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M", split="train"
        )
        print(f"OpenThoughts2-1M dataset size: {len(openthoughts_dataset)}")
    except Exception as e:
        print(f"Error loading OpenThoughts2-1M dataset: {e}")
        openthoughts_dataset = None

    # Analyze dataset structures
    if sharegpt_dataset:
        analyze_sharegpt_structure(sharegpt_dataset)
    if ultrachat_dataset:
        analyze_ultrachat_structure(ultrachat_dataset)
    if openthoughts_dataset:
        analyze_openthoughts_structure(openthoughts_dataset)

    # Perform sequence length analysis
    all_results = {}

    if sharegpt_dataset:
        sharegpt_results = analyze_dataset_lengths(sharegpt_dataset, "ShareGPT", "sharegpt", tokenizer, args.max_samples)
        if sharegpt_results:
            all_results["ShareGPT"] = sharegpt_results

    if ultrachat_dataset:
        ultrachat_results = analyze_dataset_lengths(ultrachat_dataset, "UltraChat", "ultrachat", tokenizer, args.max_samples)
        if ultrachat_results:
            all_results["UltraChat"] = ultrachat_results

    if openthoughts_dataset:
        openthoughts_results = analyze_dataset_lengths(
            openthoughts_dataset, "OpenThoughts2-1M", "openthoughts", tokenizer, args.max_samples
        )
        if openthoughts_results:
            all_results["OpenThoughts2-1M"] = openthoughts_results

    # Print results
    print("\n" + "=" * 60)
    print("SEQUENCE LENGTH ANALYSIS RESULTS")
    print("=" * 60)

    if not all_results:
        print("No valid datasets were processed successfully.")
        return

    for dataset_name, results in all_results.items():
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*40}")

        print_statistics(results["char_stats"])
        print_statistics(results["word_stats"])
        print_statistics(results["token_stats"])

    # Save results to JSON
    json_results = {}
    for dataset_name, results in all_results.items():
        json_results[dataset_name] = {
            "char_stats": results["char_stats"],
            "word_stats": results["word_stats"],
            "token_stats": results["token_stats"],
        }

    with open(f"{args.output_dir}/sequence_statistics.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nDetailed results saved to: {args.output_dir}/sequence_statistics.json")

    # # Create visualizations
    # if not args.no_plots and all_results:
    #     try:
    #         create_visualizations(all_results, f"{args.output_dir}/plots")
    #     except Exception as e:
    #         print(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    main()
