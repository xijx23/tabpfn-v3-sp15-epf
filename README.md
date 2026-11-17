# TabPFN-3 电价预测实验（CAISO SP15 日前 LMP）

从主项目中抽出的最小可运行子集，只用于在服务器 GPU 上跑 TabPFN v3 并与 v2 对比。方法依据 Lipiecki & Weron, arXiv:2609.00089（细节见 `paper_tabpfn/README.md`；其中提到的 `recent.py` 和 track1 树模型训练代码不在本仓库）。

## 环境

- Python 3.12；NVIDIA GPU，显存 ≥ 8 GB（本地 v3 峰值约 6.3 GB）。
- 先按服务器 CUDA 版本装 torch，再装其余依赖：

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 按服务器 CUDA 改 cu12x
pip install -r requirements.txt
```

- TabPFN v2.5 及以上的权重需要许可：在 https://ux.priorlabs.ai/account/licenses 接受许可后，把 API key 设为环境变量（**不要写进任何文件或提交到仓库**）：

```bash
export TABPFN_TOKEN=<你的 key>
```

## 运行

```bash
bash run_v3.sh
```

依次执行：

| 步骤 | 命令 | 说明 |
|---|---|---|
| 1 | `python -m unittest discover -s tests` | 储能策略与特征的单元测试 |
| 2 | `python -m paper_tabpfn.experiment --variant paper --variant paper_v3 --every 2` | 验证集 21 天（2024-11-23~12-06、2026-09-26~10-23 隔天取，与 testbench 不重叠），对比 v2 / v3 的中位数、均值，以及与树 ensemble 等权融合（combo）的日均 RMSE |
| 3 | `python -m paper_tabpfn.testbench_run --version v2` | v2 全部读缓存，不占 GPU，作为对照 |
| 4 | `python -m paper_tabpfn.testbench_run --version v3` | testbench 三区间（w7_2026、d14_2026、d14_2025，共 28 个不同日期） |

耗时参考（RTX 5060 8 GB）：v3 每天约 5.5 分钟（24 个小时模型 × 12–16 秒）。步骤 2 还剩 6 天（已附 15 天缓存），步骤 4 约 28 天，合计约 3 小时。

预测按天写入缓存，中断后重跑同一命令会跳过已完成的日期。

## 输出

- `logs/`：各步骤日志，`experiment_v3.log` 末尾是验证集对比表，`testbench_v*.log` 是各区间 RMSE 与储能收益。
- `paper_tabpfn/reports/forecasts_v3.csv`：v3 testbench 预测（均值、中位数、11 个分位数）。
- `paper_tabpfn/reports/experiments/paper_v3.csv`：v3 验证集逐小时预测。
- `paper_tabpfn/reports/<interval>_v3/`：v3 的提交矩阵；`testbench/results/<interval>/t1_tabpfn_v3*`：评分结果。

## 数据

- `data/processed/hourly_wide.parquet`：CAISO OASIS 公开数据整理的小时宽表（截至 2026-11-17），为修订版历史值，不是截止时刻快照。
- `testbench/truth/`：三个区间的冻结真值（含 sha256 校验）。
- `track1/reports/...`：树模型 ensemble / ensemble_analog 的已有预测，只用于 combo 融合对照。

信息截止口径：D 日预测只用 D-1 及更早的价格、D 日 2DA 负荷与风光预测、D-1 燃料与碳价。
