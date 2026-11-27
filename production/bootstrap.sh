#!/usr/bin/env bash
set -euo pipefail
cd /content
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ "$gpu" == *"RTX PRO 6000"* && "$gpu" == *Blackwell* ]] || { echo "Unexpected GPU: $gpu"; exit 1; }
rev=890f4211af7e15c5e904f6a105c7fee877340919
if [[ ! -d minimax-h3/.git ]]; then
    git clone --filter=blob:none https://github.com/keboqi/minimax-h3.git minimax-h3
fi
git -C minimax-h3 checkout --detach "$rev"
if ! command -v uv >/dev/null; then python3 -m pip install uv; fi
if [[ ! -x /content/venv_h3/bin/python ]]; then
    uv venv --seed --python 3.12 /content/venv_h3
fi
source /content/venv_h3/bin/activate
cd /content/minimax-h3
python setup_h3.py --install-dir /content/minimax-h3/h3 --skip-cjk-font
cd h3/ComfyUI
exec python -u main.py --listen 127.0.0.1 --port 8188 --fast fp16_accumulation --use-ck-attention
