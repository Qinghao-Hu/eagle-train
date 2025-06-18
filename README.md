<h1 style="text-align: center;">Eagle-Train</h1>

# Installation

```bash
git clone https://github.com/Qinghao-Hu/eagle-train.git
conda create --name eagle python=3.10
conda activate eagle
pip install -e .
```

Install flash_attn
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```





# Step 1: Create Mixed DataSet

Download processed dataset:

```bash
huggingface-cli download Qinghao/eagle-mix --repo-type dataset --local-dir /path/to/your/directory
```

[Skip following]

## Dataset Statistics

Create mixed dataset from the following datasets:

| Dataset | Count | Mean | Median | Max |
|---------|--------|-------|---------|-----|
| ShareGPT | 68,623 | 6,128 | 6,445 | 93,262 |
| UltraChat | 207,865 | 5,686 | 5,230 | 53,213 |
| OpenThoughts2-1M | 1,143,205 | 16,175 | 10,859 | 996,361 |

```bash
python create_dataset/create_mixed_dataset.py
```

<!-- /nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M
/nobackup/qinghao/trace/ShareGPT_V4.3_unfiltered_cleaned_split.json
/nobackup/qinghao/dataset/ultrachat_200k -->


# Step 2: Generate Frequency Mapping
[Skip following]
Generate `d2t` and `t2d` mapping for the given tokenizer and mixed dataset.

```bash
python freq_map/generate_freq.py
```

# Step 3: Cache Hidden States

Cache hidden states. Typically, takes around 6~8 hours for 1 node. Need storage space >50 TB.

```bash
srun -J datagen -N 1 --exclusive bash scripts/datagen.sh
```


# Step 4: Train

You can change -N to modify the number of nodes.

```bash
srun -J eagle3 -N 2 --exclusive bash scripts/train_eagle3.sh
```

# Models

- [ ] Llama-3.1-8B-Instruct
- [ ] Qwen2.5-1.5B-Instruct
- [ ] Qwen2.5-3B-Instruct
- [ ] Qwen2.5-7B-Instruct
- [ ] Qwen2.5-14B-Instruct
- [ ] Qwen2.5-1.5B
- [ ] Qwen2.5-3B
- [ ] Qwen2.5-7B
- [ ] Qwen2.5-14B
- [ ] Qwen3-1.7B
- [ ] Qwen3-4B
- [ ] Qwen3-8B
- [ ] Qwen3-14B
- [ ] Qwen3-1.7B-Base
- [ ] Qwen3-4B-Base
- [ ] Qwen3-8B-Base
- [ ] Qwen3-14B-Base


Large models:

- [ ] Llama-3.3-70B-Instruct
- [ ] Qwen2.5-32B-Instruct
- [ ] Qwen2.5-72B-Instruct
- [ ] Qwen2.5-32B
- [ ] Qwen2.5-72B
- [ ] Qwen3-32B

