# paper_tabpfn：TabPFN 电价预测 + 论文储能策略

依据 Lipiecki & Weron, *Foundation models for electricity price forecasting and battery arbitrage*（arXiv:2609.00089）。论文中统计精度最好的是 TabPFN 系列，这里用 TabPFN-2（零样本、上下文学习，不训练权重）。独立模块，不改 track1/track2 的代码，只以只读方式调用 `track2.physics` / `track2.config` 和 testbench。

## 依赖

`requirements.txt` 之外另装（RTX 50 系需要 cu128 版 torch）：

```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install tabpfn
```

已验证版本：torch 2.11.0+cu128、tabpfn 10.0.0。首次运行会从 HuggingFace 下载 TabPFN-2 回归权重。

## 模型（论文 Eq. 5，§4.2.1）

每个目标日 d、每个小时 h 一个 TabPFN 回归，一天 24 次。特征（一天一行，共 243 列）：

| 论文 | 本项目对应 |
|---|---|
| p_{d-1}, p_{d-2}, p_{d-3}, p_{d-7}（各 24 小时） | SP15 DAM LMP |
| L_d, L_{d-1}, L_{d-7} | CAISO 2DA 负荷预测 `load_fc_2da__CA ISO-TAC__2` |
| R_d, R_{d-1}, R_{d-7} | NP15/SP15/ZP26 光伏+风电 DAM 预测之和，缺失分量按论文记 0 |
| EUA/TTF/Brent/API2（d-2） | SoCal 气价 FRSCE1、碳价，按项目截止口径取 d-1 |
| DoW | 星期 |

- 上下文：d 之前最近 1449 天（论文值；实际可用历史不足，d14_2025 约 836 天，2026 区间约 1170 天），每个目标日滚动更新，只用 d 之前的数据。
- 点预测取预测分布中位数（论文 Eq. 10），提交为 `t1_tabpfn`；均值另交 `t1_tabpfn_mean` 作对照。
- 秋季 DST 日保留第一个 01:00，春季 23 小时日不进上下文。
- GPU 上一天约 2.5 分钟；预测缓存在 `reports/forecasts.csv`（含均值、中位数和 11 个分位数），重跑不会重复计算。

## 储能策略（论文 §6.2，按赛题改编）

论文：1 MWh 电池，每天选一个买入小时 h_b < 卖出小时 h_s，使 `η·p̂_hs − p̂_hb/η − C` 最大，预测利润为正才交易。

- **UB（`t2_tabpfn_ub`）**：7 MW、可用 12.6 MWh 一小时充不满，所以把一对买/卖小时推广为每天一个循环：在分割点之前选最便宜的小时充电，之后选最贵的小时放电，逐段配对直到边际利润 ≤ 0；24 个分割点都试一遍，取预测利润最大的那个，不为正就不交易。前一天剩下的电量视为零成本，优先在高价小时放掉。C 取 0（赛题不计循环成本）。
- **QB α（`t2_tabpfn_qb90/70/50`）**：论文用分位数做限价的环形报价，有条件成交；赛题只能交固定功率，所以改成开不开仓的过滤：UB 方案按悲观价（卖出按 q^τ、买入按 q^{1−τ}，τ=(1−α)/2）重新计价，仍为正才执行，否则当天全天不动。注意这里 α 越大越保守，和论文里“α 越大风险偏好越高”方向相反。
- 输出先经 `track2.physics.quantize_schedule` 量化成可行解，SOC 按实际价格结算后逐日接续，每个区间都从 5% 起算。

## 结果分析与改进（2026-11-15）

### 电价

- **TabPFN 与树模型互补**：两者赢的日子不一样。验证窗口（2024-11-23~12-06、2026-09-26~10-23，隔天取 21 天，与 testbench 不重叠，`python -m paper_tabpfn.experiment`）上，TabPFN 日均 RMSE 4.08，树 ensemble（track1 逐日重训滚动）5.41；testbench 上则反过来。单一模型跨窗口都不稳定。
- TabPFN 的误差集中在光伏时段 HE8–15：午间低谷预测得过深（11-01 实际约 28，预测 2–8）。树模型的误差主要在晚高峰。
- **特征变体都没赢过论文原版**（验证集，中位数日均 RMSE）：每小时约 20 列的紧凑特征 4.69、上下文缩短到 365 天 5.71、以昨日同小时为基准的差分目标 4.78，论文 243 列原版 4.08。保留原版。
- **改进：与树 ensemble_analog 等权平均（`t1_tabpfn_combo`）**。权重事先定为 0.5，没有调。日均 RMSE 3.650 / 3.971 / 3.455（w7 / d14_2026 / d14_2025），三个区间都优于树 ensemble_analog 本身（3.693 / 4.008 / 3.563）；s=20 相对分在三个区间都排第 1。最优权重在 testbench（0.25–0.4）和验证集（0.75）之间不稳定，所以不调权重。

### 储能

- 论文的单循环规则是策略本身的上限：用实际价格（完美预知）也只能拿到 2722 / 3808 / 3122，与现有最佳预测型策略相当。日内允许多循环（DP 精确求解）在完美预知下与单循环一样。
- 缺的是跨日：48 小时视野的完美预知上限是 2910 / 4247 / 3464，即把电留到次日早高峰。区间最后一天必须缩回 24 小时，否则留到区间外的电不计分（完美预知下反而降到 2323）。
- **改进：`t2_tabpfn_dp48`**，在 TabPFN 中位数上做 DP，D+1 用最近 4 周同星期同小时均价补齐（与 track2 相同，只读调用 `track2.scenarios`），最后一天 24 小时。收益 2710 / 3858 / 3259，合计 9827，高于现有任一单个策略的合计（最好的是 teammate1_method1，9738）。延续方式“重复当日预测”合计 9339，更差；这个对比是在 testbench 上做的，有选择偏差。
- **注意评分口径**：dp48 总收益高，但逐日相对分只排中游（w7 第 24/34）。赛题按天在所有参赛者之间横向打分，跨日持有电量会让充电当天的收益偏低，所以“总收益最大”不等于“相对分最大”。
- `t1_tabpfn_combo` 和 `t2_tabpfn_combo_dp48` 读取 `track1/reports/testbench/<interval>/price_ensemble_analog.csv`，需要先跑 `track1.testbench_run`。

### 未做 / 待定

- TabPFN v2.5、v2.6、v3 的权重需要在浏览器里接受许可后才能下载，论文里 TabPFN-3 略好于 v2。是否接受许可由使用者决定。
- 针对按天相对分的储能目标，例如对跨日持有加惩罚，还没有做。

## 运行

```powershell
.venv\Scripts\python -m paper_tabpfn.testbench_run                  # 3 个区间
.venv\Scripts\python -m paper_tabpfn.testbench_run --interval d14_2025
.venv\Scripts\python -m unittest tests.test_paper_tabpfn -v
```

输出 `reports/<interval>/price_*.csv`、`dispatch_*.csv`，同时以上面的名字提交到 `testbench/results/<interval>/`。

```powershell
.venv\Scripts\python -m paper_tabpfn.experiment --variant paper --variant compact --every 2   # 验证集变体对比
```
