#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
port_venv=${1:-"$here/.venv"}
python3 -m venv "$port_venv"
"$port_venv/bin/python" -m pip install --upgrade pip
"$port_venv/bin/python" -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
"$port_venv/bin/python" -m pip install -r "$here/requirements-tpu.txt" -f https://storage.googleapis.com/libtpu-releases/index.html
"$port_venv/bin/python" -c 'import torch,torch_xla,tiktoken; print(torch.__version__,torch_xla.__version__); tiktoken.get_encoding("gpt2")'
