# testbench：统一评价

在项目根目录运行，Windows 用 `.venv\Scripts\python`。

## 测试区间（加州运行日，首尾都包含）

| interval | 日期 | 天数 N |
|---|---|---|
| `w7_2026` | 2026-11-02 至 11-08 | 7 |
| `d14_2026` | 2026-10-26 至 11-08 | 14 |
| `d14_2025` | 2025-11-23 至 12-06 | 14 |

## 提交文件

每个区间交两个**无表头、N 行 × 24 列的纯数字 CSV**：

- 电价：SP15 DAM 总 LMP 预测，单位 USD/MWh。
- 储能：净功率，单位 MW，正值放电、负值充电。

第 i 行是区间第 i 天，第 j 列是本地时间 j:00–(j+1):00（j = 0…23）。2026-11-01 的第 2 列是第一个 01:00（PDT），第二个 01:00 不计入。

注意：

- 储能每个区间第一天都从 SOC 5% 起算，所以 `w7_2026` 必须单独提交，不能从 `d14_2026` 的文件里截取。
- 只参加一个赛道的，只交对应的那个文件。
- 形状不对、有空值或非数字的文件会被整体拒收，不评分。

## 真值

```powershell
.venv\Scripts\python -m testbench.truth                      # 生成全部区间
.venv\Scripts\python -m testbench.truth --interval d14_2026 --force   # 强制重建
```

- 真值从 `data/processed/hourly_wide.parquet` 生成，并与 `data/raw/dam_lmp` 中的 OPR_DT/OPR_HR 逐值对照（OPR_HR k 对应第 k-1 列，OPR_HR 25 为第二个 01:00，丢弃）。
- 输出 `truth/<interval>.csv` 和 `.json`（含 sha256）。真值生成后即冻结，已存在时拒绝覆盖。
- 评价时会校验真值的哈希，结果里也记录了真值哈希，阶段二只比较用同一份真值算出来的结果。

## 阶段一：各自的指标

```powershell
.venv\Scripts\python -m testbench.evaluate --interval d14_2025 --name alice_xgb --price price.csv --dispatch dispatch.csv
```

输出到 `results/<interval>/<name>/`。同名重交会覆盖旧结果。

| 文件 | 内容 |
|---|---|
| `price.csv`、`dispatch.csv` | 提交原件 |
| `price_daily.csv` | date, rmse（每日 RMSE） |
| `storage_daily.csv` | date, revenue, valid, reason, soc_start, soc_end |
| `summary.json` | 平均每日 RMSE、区间累计收益、违规天数、提交和真值的 sha256 |

储能规则直接复用 `track2/physics.py`，与手册一致：

- 参数：14 MWh、7 MW，SOC 5%–95%，单程效率 85%；
- SOC 跨日连续，期末不约束；
- **任一小时超功率或 SOC 越界，当天全天不充不放、收益为 0、SOC 保持当日初始值**，`reason` 写明是哪一小时违反了哪条约束。

## 阶段二：相对分（等所有人都交完后运行，可重复运行）

```powershell
.venv\Scripts\python -m testbench.relative --interval d14_2025
```

- 只在 `results/<interval>/` 下已有的团队模型之间比较，不设基线。
- 每天按手册公式计算：`clip(60 + s·(x − cross_mean)/cross_std, 0, 100)`。
  - 电价取 x = −RMSE，也就是 RMSE 越低越好；储能取 x = 收益。
  - `cross_mean`、`cross_std` 是当天所有模型之间的横截面均值和总体标准差（ddof=0）；`cross_std = 0` 时所有模型记 60 分。
- s 同时算 10 和 20 两套。

输出 `results/<interval>/_relative/`：

| 文件 | 内容 |
|---|---|
| `price_relative_daily.csv`、`storage_relative_daily.csv` | 每天各模型的 s10、s20 相对分，以及当天的 cross_mean、cross_std |
| `relative_summary.csv` | 每个模型、赛道、s 的区间平均相对分，以及参与比较的模型列表 |

只有 2 个模型时，每天必然一个是 60+s、一个是 60−s，相对分只反映当天谁赢，不反映赢了多少。

## 不检查的内容

是否使用了未来数据、训练集是否早于测试区间，由成员自己保证，testbench 不检查。
