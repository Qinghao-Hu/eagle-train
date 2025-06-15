import datasets
from collections import Counter


def test_eagle_mix_dataset():
    print("Loading eagle-mix dataset...")

    # Load from disk
    eagle_mix_dataset = datasets.load_from_disk("/nobackup/qinghao/dataset/eagle-mix")

    print(f"Total eagle-mix dataset size: {len(eagle_mix_dataset):,}")
    print(f"Dataset features: {eagle_mix_dataset.features}")

    # Count by source
    print("\nDataset composition by source:")
    sources = [example["source"] for example in eagle_mix_dataset]
    source_counts = Counter(sources)

    for source, count in source_counts.items():
        percentage = (count / len(eagle_mix_dataset)) * 100
        print(f"{source}: {count:,} samples ({percentage:.1f}%)")

    # Show sample examples from each source
    print("\nSample examples from each source:")

    for source in source_counts.keys():
        print(f"\n{'='*50}")
        print(f"Sample from {source}:")
        print(f"{'='*50}")

        # Find first example from this source
        sample = next(ex for ex in eagle_mix_dataset if ex["source"] == source)

        print(f"Available keys: {list(sample.keys())}")
        print(f"Source: {sample['source']}")
        print(f"Number of conversation turns: {len(sample['conversations'])}")

        # Show first few conversation turns
        for i, conv in enumerate(sample["conversations"][:3]):  # Show first 3 turns
            print(f"Turn {i+1}:")
            print(f"  From: {conv['from']}")
            print(f"  Value: {conv['value'][:200]}..." if len(conv["value"]) > 200 else f"  Value: {conv['value']}")

        if len(sample["conversations"]) > 3:
            print(f"  ... and {len(sample['conversations']) - 3} more turns")

    print(f"\n{'='*50}")
    print("Eagle-mix dataset verification completed!")
    print(f"{'='*50}")


if __name__ == "__main__":
    test_eagle_mix_dataset()
