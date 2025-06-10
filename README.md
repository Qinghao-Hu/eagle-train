<h1 style="text-align: center;">Eagle-Train</h1>

# Installation

```bash
git clone --recursive https://github.com/mit-han-lab/eagle-train.git
conda create --name eagle python=3.10
conda activate eagle
pip install -e .
```

Install liger-kernel and flash_attn
```bash
pip install liger-kernel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```