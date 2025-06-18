# Eagle-Mix Dataset

## Dataset Description

Eagle-Mix is a comprehensive mixed dataset created for training Eagle models. It combines high-quality conversational data from multiple sources to provide diverse training examples.

## Dataset Composition

The dataset is composed of the following sources:

| Dataset | Count | Mean Length | Median Length | Max Length |
|---------|--------|-------------|---------------|-------------|
| ShareGPT | 68,623 | 6,128 | 6,445 | 93,262 |
| UltraChat | 207,865 | 5,686 | 5,230 | 53,213 |
| OpenThoughts2-1M | 1,143,205 | 16,175 | 10,859 | 996,361 |

**Mixture**: Full 68K ShareGPT, Full 200K UltraChat, Sampled 250k OpenThoughts2

## Dataset Features

The dataset contains conversational data with the following structure:
- **conversations**: Multi-turn conversations between users and assistants
- **source**: Original dataset source identifier
- **quality_score**: Quality assessment score (if available)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Qinghao/eagle-mix")

# Example usage
for example in dataset['train']:
    print(example['conversations'])
```
