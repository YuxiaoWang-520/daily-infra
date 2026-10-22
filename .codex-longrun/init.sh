#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Use installed dependencies only. Each article runs in a fresh interpreter.
python3 -c 'import sys, torch; print(sys.version.split()[0], torch.__version__, "CUDA:", torch.cuda.is_available())'
for article in wiki/2026/10/*; do
  if [ -d "$article/code" ]; then
    python3 -m unittest discover -s "$article/code" -p 'test_*.py' -v
  fi
done
git diff --check
