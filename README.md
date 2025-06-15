<h1 style="text-align: center;">Eagle-Train</h1>

# Installation

```bash
git clone --recursive https://github.com/mit-han-lab/eagle-train.git
conda create --name eagle python=3.10
conda activate eagle
pip install -e .
```

Install flash_attn
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```





# Step 1: Create Mixed DataSet

Create mixed dataset from the following datasets:

- ShareGPT-V4.3
- UltraChat-200k
- OpenThoughts2-1M

```bash
python create_dataset/create_mixed_dataset.py
```

/nobackup/qinghao/dataset/reasoning/OpenThoughts2-1M
/nobackup/qinghao/trace/ShareGPT_V4.3_unfiltered_cleaned_split.json
/nobackup/qinghao/dataset/ultrachat_200k


New dataset:
/nobackup/qinghao/dataset/eagle-mix

# Step 2: Generate Frequency Mapping

Generate `d2t` and `t2d` mapping for the given tokenizer and mixed dataset.

```bash
python freq_map/generate_freq.py
```

# Step 3: Cache Hidden States

Cache hidden states .

```bash
python cache_hidden_states.py
```


# Models

- Qwen2.5-7B-Instruct
- Qwen2.5-Base-7B
- Qwen2.5-32B
- Llama-3.3-70B-Instruct
- Llama-3-8B-Instruct
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-72B