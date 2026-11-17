#!/usr/bin/env bash
# 一键：单元测试 -> 验证集 v2 vs v3 -> testbench v2（读缓存）与 v3。中断后重跑会从缓存续跑。
set -euo pipefail
cd "$(dirname "$0")"
: "${TABPFN_TOKEN:?请先 export TABPFN_TOKEN=<你的 Prior Labs API key>}"
export TABPFN_NO_BROWSER=1
export PYTHONIOENCODING=utf-8
PY="${PYTHON:-python}"
mkdir -p logs
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用：v3 在 CPU 上极慢'; print('GPU:', torch.cuda.get_device_name(0))"
"$PY" -m unittest discover -s tests -v 2>&1 | tee logs/tests.log
"$PY" -m paper_tabpfn.experiment --variant paper --variant paper_v3 --every 2 2>&1 | tee logs/experiment_v3.log
"$PY" -m paper_tabpfn.testbench_run --version v2 2>&1 | tee logs/testbench_v2.log
"$PY" -m paper_tabpfn.testbench_run --version v3 2>&1 | tee logs/testbench_v3.log
echo "全部完成。结果见 logs/ 与 paper_tabpfn/reports/。"
