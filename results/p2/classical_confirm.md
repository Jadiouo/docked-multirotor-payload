# 古典控制器 confirmatory seed 重算（prereg §14(a)）

- 來源腳本：`scripts/classical_confirm.py`；每格 100 episode；confirmatory seed offset 900000（env seed = SEED[task] + 900000；event 用 SEED_EV[task] + 900000）。
- 控制器參數：g2_estim_tune.json best（estim 一律 snap=False）、g1_coord_tune.json best（yawlim）、rqd_tune.json best（重置偵測）；與探索 seed 相同，未重調。
- 分數 = success x max(0, 1 - e_rms/e_ref)；區間 = 對 episode 重抽 2000 次的 95% bootstrap 百分位區間（均值）。差的區間為逐 episode 成對差的 bootstrap。
- 探索 seed 舊值取自 `results/p2/` 下 p3d_factorial_raw.json（含 snap=False 的 `n` 版）、g2_raw.json、g2_informed_raw.json、rqd_classical_raw.json（欄位『出處』）；沒有舊值者標『無舊值』。
- 方向規則（機械）：對比（A-B）的 confirmatory 差與探索差，|差| < 0.002 視為 0；符號相同記『一致』，否則『不一致』，缺舊值記 n/a。
- RQ-D 的『落在 ±B% 內的比例』：B 在 prereg 未填（P4 未開），此報告不計算；只給每格估計值、CI 與成對差。
- E2 的 informed 參考不支援（需逐 episode host），E2 只列 B1、B1-QP、B3 系列、estim-yawlim。


## E1：payload15（15%，argmax(x) 模組中心，t=0）

| 構型 | 任務 | 方法 | confirmatory 分數 [95% CI] | 成功率 | 探索 seed 分數 | 出處 |
|---|---|---|---|---|---|---|
| line6 | T2 | B1 | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP | 0.011 [0.002, 0.023] | 6% | 0.002 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-yawlim | 0.021 [0.008, 0.037] | 11% | 0.012 | p3d_factorial_raw.json |
| line6 | T2 | B3 INDI | 0.224 [0.185, 0.266] | 64% | 0.176 | g2_raw.json |
| line6 | T2 | B3+adapt | 0.219 [0.179, 0.259] | 65% | 0.178 | g2_raw.json |
| line6 | T2 | B1-QP-informed | 0.262 [0.223, 0.301] | 76% | 0.210 | g2_informed_raw.json |
| line6 | T2 | B1-QP-informed-yawlim | 0.341 [0.307, 0.373] | 94% | 0.308 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-estim | 0.257 [0.217, 0.296] | 73% | 0.211 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-estim-yawlim | 0.340 [0.306, 0.372] | 94% | 0.307 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1 | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP | 0.104 [0.071, 0.140] | 30% | 0.119 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-yawlim | 0.107 [0.074, 0.144] | 30% | 0.121 | p3d_factorial_raw.json |
| line6 | POS_STEP | B3 INDI | 0.616 [0.613, 0.620] | 100% | 0.616 | g2_raw.json |
| line6 | POS_STEP | B3+adapt | 0.605 [0.601, 0.609] | 100% | 0.604 | g2_raw.json |
| line6 | POS_STEP | B1-QP-informed | 0.636 [0.634, 0.637] | 100% | 0.635 | g2_informed_raw.json |
| line6 | POS_STEP | B1-QP-informed-yawlim | 0.635 [0.633, 0.637] | 100% | 0.635 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-estim | 0.633 [0.631, 0.635] | 100% | 0.633 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-estim-yawlim | 0.633 [0.631, 0.635] | 100% | 0.632 | p3d_factorial_raw.json |
| line7 | T2 | B1 | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP | 0.010 [0.000, 0.022] | 3% | 0.001 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-yawlim | 0.011 [0.001, 0.023] | 6% | 0.007 | p3d_factorial_raw.json |
| line7 | T2 | B3 INDI | 0.181 [0.140, 0.223] | 52% | — | 無舊值 |
| line7 | T2 | B3+adapt | 0.182 [0.144, 0.222] | 55% | — | 無舊值 |
| line7 | T2 | B1-QP-informed | 0.228 [0.188, 0.267] | 68% | — | 無舊值 |
| line7 | T2 | B1-QP-informed-yawlim | 0.311 [0.274, 0.345] | 87% | 0.291 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-estim | 0.229 [0.190, 0.266] | 68% | 0.172 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-estim-yawlim | 0.310 [0.275, 0.344] | 88% | 0.289 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1 | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | POS_STEP | B3 INDI | 0.590 [0.582, 0.599] | 100% | — | 無舊值 |
| line7 | POS_STEP | B3+adapt | 0.569 [0.557, 0.580] | 100% | — | 無舊值 |
| line7 | POS_STEP | B1-QP-informed | 0.633 [0.631, 0.635] | 100% | — | 無舊值 |
| line7 | POS_STEP | B1-QP-informed-yawlim | 0.631 [0.629, 0.633] | 100% | 0.630 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-estim | 0.630 [0.628, 0.632] | 100% | 0.628 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-estim-yawlim | 0.628 [0.625, 0.630] | 100% | 0.626 | p3d_factorial_raw.json |
| line8 | T2 | B1 | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP | 0.000 [0.000, 0.001] | 1% | 0.000 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-yawlim | 0.009 [0.000, 0.021] | 3% | 0.000 | p3d_factorial_raw.json |
| line8 | T2 | B3 INDI | 0.155 [0.115, 0.196] | 45% | 0.111 | g2_raw.json |
| line8 | T2 | B3+adapt | 0.137 [0.100, 0.175] | 45% | 0.078 | g2_raw.json |
| line8 | T2 | B1-QP-informed | 0.192 [0.151, 0.233] | 57% | 0.147 | g2_informed_raw.json |
| line8 | T2 | B1-QP-informed-yawlim | 0.289 [0.252, 0.326] | 81% | 0.255 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-estim | 0.192 [0.151, 0.233] | 58% | 0.151 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-estim-yawlim | 0.283 [0.245, 0.319] | 79% | 0.255 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1 | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | POS_STEP | B3 INDI | 0.457 [0.411, 0.496] | 84% | 0.487 | g2_raw.json |
| line8 | POS_STEP | B3+adapt | 0.440 [0.397, 0.481] | 82% | 0.478 | g2_raw.json |
| line8 | POS_STEP | B1-QP-informed | 0.621 [0.618, 0.624] | 100% | 0.624 | g2_informed_raw.json |
| line8 | POS_STEP | B1-QP-informed-yawlim | 0.616 [0.613, 0.619] | 100% | 0.619 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-estim | 0.615 [0.611, 0.618] | 100% | 0.618 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-estim-yawlim | 0.610 [0.605, 0.614] | 100% | 0.613 | p3d_factorial_raw.json |

### E1 成對對比（A - B；逐 episode 成對）

| 構型 | 任務 | A | B | confirmatory 差 [95% CI] | A/B 比 | 探索 seed 差 | 方向 |
|---|---|---|---|---|---|---|---|
| line6 | T2 | B1-QP-informed-yawlim | B1-QP-estim-yawlim | +0.001 [-0.001, +0.003] | 1.003 | +0.001 | 一致 |
| line6 | T2 | B1-QP-estim-yawlim | B1-QP-estim | +0.083 [+0.057, +0.112] | 1.322 | +0.096 | 一致 |
| line6 | T2 | B1-QP-estim-yawlim | B1-QP-yawlim | +0.319 [+0.286, +0.351] | — | +0.296 | 一致 |
| line6 | POS_STEP | B1-QP-informed-yawlim | B1-QP-estim-yawlim | +0.002 [+0.002, +0.003] | 1.004 | +0.002 | 一致 |
| line6 | POS_STEP | B1-QP-estim-yawlim | B1-QP-estim | -0.001 [-0.001, -0.000] | 0.999 | -0.001 | 一致 |
| line6 | POS_STEP | B1-QP-estim-yawlim | B1-QP-yawlim | +0.526 [+0.488, +0.559] | 5.904 | +0.511 | 一致 |
| line7 | T2 | B1-QP-informed-yawlim | B1-QP-estim-yawlim | +0.001 [-0.006, +0.005] | 1.002 | +0.001 | 一致 |
| line7 | T2 | B1-QP-estim-yawlim | B1-QP-estim | +0.081 [+0.056, +0.110] | 1.356 | +0.117 | 一致 |
| line7 | T2 | B1-QP-estim-yawlim | B1-QP-yawlim | +0.300 [+0.265, +0.332] | — | +0.282 | 一致 |
| line7 | POS_STEP | B1-QP-informed-yawlim | B1-QP-estim-yawlim | +0.003 [+0.003, +0.004] | 1.005 | +0.004 | 一致 |
| line7 | POS_STEP | B1-QP-estim-yawlim | B1-QP-estim | -0.002 [-0.002, -0.002] | 0.997 | -0.002 | 一致 |
| line7 | POS_STEP | B1-QP-estim-yawlim | B1-QP-yawlim | +0.628 [+0.625, +0.630] | — | +0.626 | 一致 |
| line8 | T2 | B1-QP-informed-yawlim | B1-QP-estim-yawlim | +0.006 [-0.005, +0.019] | 1.021 | +0.000 | 不一致 |
| line8 | T2 | B1-QP-estim-yawlim | B1-QP-estim | +0.091 [+0.060, +0.127] | 1.477 | +0.104 | 一致 |
| line8 | T2 | B1-QP-estim-yawlim | B1-QP-yawlim | +0.274 [+0.235, +0.310] | — | +0.254 | 一致 |
| line8 | POS_STEP | B1-QP-informed-yawlim | B1-QP-estim-yawlim | +0.007 [+0.006, +0.008] | 1.011 | +0.006 | 一致 |
| line8 | POS_STEP | B1-QP-estim-yawlim | B1-QP-estim | -0.005 [-0.006, -0.004] | 0.992 | -0.005 | 一致 |
| line8 | POS_STEP | B1-QP-estim-yawlim | B1-QP-yawlim | +0.610 [+0.605, +0.614] | — | +0.613 | 一致 |

## E2：留出集 RandomPayloadEnv（質量固定 x 每 episode 隨機掛點）

| 構型 | 任務 | 質量 | 方法 | confirmatory 分數 [95% CI] | 成功率 | 探索 seed 分數 | 出處 |
|---|---|---|---|---|---|---|---|
| line6 | T2 | 0.10 | B1 | 0.160 [0.122, 0.196] | 51% | 0.106 | p3d_factorial_raw.json |
| line6 | T2 | 0.10 | B1-QP | 0.238 [0.200, 0.276] | 70% | 0.216 | p3d_factorial_raw.json |
| line6 | T2 | 0.10 | B3 INDI | 0.317 [0.281, 0.352] | 86% | — | 無舊值 |
| line6 | T2 | 0.10 | B3+adapt | 0.310 [0.272, 0.345] | 86% | — | 無舊值 |
| line6 | T2 | 0.10 | B1-QP-estim-yawlim | 0.403 [0.376, 0.428] | 99% | 0.374 | p3d_factorial_raw.json |
| line6 | T2 | 0.15 | B1 | 0.062 [0.038, 0.089] | 22% | 0.039 | p3d_factorial_raw.json |
| line6 | T2 | 0.15 | B1-QP | 0.154 [0.119, 0.189] | 53% | 0.094 | p3d_factorial_raw.json |
| line6 | T2 | 0.15 | B3 INDI | 0.272 [0.231, 0.310] | 77% | — | 無舊值 |
| line6 | T2 | 0.15 | B3+adapt | 0.273 [0.233, 0.312] | 78% | — | 無舊值 |
| line6 | T2 | 0.15 | B1-QP-estim-yawlim | 0.372 [0.340, 0.402] | 94% | 0.352 | p3d_factorial_raw.json |
| line6 | T2 | 0.20 | B1 | 0.016 [0.005, 0.029] | 12% | 0.009 | p3d_factorial_raw.json |
| line6 | T2 | 0.20 | B1-QP | 0.057 [0.035, 0.080] | 22% | 0.034 | p3d_factorial_raw.json |
| line6 | T2 | 0.20 | B3 INDI | 0.222 [0.180, 0.264] | 61% | — | 無舊值 |
| line6 | T2 | 0.20 | B3+adapt | 0.210 [0.169, 0.251] | 58% | — | 無舊值 |
| line6 | T2 | 0.20 | B1-QP-estim-yawlim | 0.281 [0.238, 0.322] | 74% | 0.234 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.10 | B1 | 0.436 [0.386, 0.484] | 75% | 0.463 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.10 | B1-QP | 0.473 [0.427, 0.515] | 87% | 0.500 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.10 | B3 INDI | 0.638 [0.637, 0.640] | 100% | — | 無舊值 |
| line6 | POS_STEP | 0.10 | B3+adapt | 0.636 [0.635, 0.638] | 100% | — | 無舊值 |
| line6 | POS_STEP | 0.10 | B1-QP-estim-yawlim | 0.641 [0.640, 0.642] | 100% | 0.640 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.15 | B1 | 0.289 [0.236, 0.338] | 54% | 0.298 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.15 | B1-QP | 0.340 [0.291, 0.386] | 67% | 0.353 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.15 | B3 INDI | 0.631 [0.628, 0.634] | 100% | — | 無舊值 |
| line6 | POS_STEP | 0.15 | B3+adapt | 0.624 [0.620, 0.628] | 100% | — | 無舊值 |
| line6 | POS_STEP | 0.15 | B1-QP-estim-yawlim | 0.638 [0.636, 0.640] | 100% | 0.637 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.20 | B1 | 0.157 [0.117, 0.198] | 38% | 0.142 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.20 | B1-QP | 0.245 [0.200, 0.288] | 55% | 0.251 | p3d_factorial_raw.json |
| line6 | POS_STEP | 0.20 | B3 INDI | 0.521 [0.477, 0.560] | 87% | — | 無舊值 |
| line6 | POS_STEP | 0.20 | B3+adapt | 0.524 [0.482, 0.563] | 87% | — | 無舊值 |
| line6 | POS_STEP | 0.20 | B1-QP-estim-yawlim | 0.594 [0.576, 0.610] | 100% | 0.598 | p3d_factorial_raw.json |
| line7 | T2 | 0.10 | B1 | 0.116 [0.082, 0.152] | 36% | 0.079 | p3d_factorial_raw.json |
| line7 | T2 | 0.10 | B1-QP | 0.210 [0.171, 0.250] | 66% | 0.155 | p3d_factorial_raw.json |
| line7 | T2 | 0.10 | B3 INDI | 0.282 [0.240, 0.322] | 76% | — | 無舊值 |
| line7 | T2 | 0.10 | B3+adapt | 0.291 [0.251, 0.328] | 80% | — | 無舊值 |
| line7 | T2 | 0.10 | B1-QP-estim-yawlim | 0.389 [0.360, 0.416] | 97% | 0.355 | p3d_factorial_raw.json |
| line7 | T2 | 0.15 | B1 | 0.045 [0.023, 0.070] | 14% | 0.025 | p3d_factorial_raw.json |
| line7 | T2 | 0.15 | B1-QP | 0.133 [0.101, 0.167] | 45% | 0.089 | p3d_factorial_raw.json |
| line7 | T2 | 0.15 | B3 INDI | 0.251 [0.210, 0.292] | 70% | — | 無舊值 |
| line7 | T2 | 0.15 | B3+adapt | 0.238 [0.197, 0.279] | 70% | — | 無舊值 |
| line7 | T2 | 0.15 | B1-QP-estim-yawlim | 0.352 [0.317, 0.386] | 90% | 0.319 | p3d_factorial_raw.json |
| line7 | T2 | 0.20 | B1 | 0.011 [0.003, 0.021] | 6% | 0.005 | p3d_factorial_raw.json |
| line7 | T2 | 0.20 | B1-QP | 0.053 [0.032, 0.077] | 22% | 0.030 | p3d_factorial_raw.json |
| line7 | T2 | 0.20 | B3 INDI | 0.173 [0.133, 0.215] | 51% | — | 無舊值 |
| line7 | T2 | 0.20 | B3+adapt | 0.156 [0.118, 0.195] | 46% | — | 無舊值 |
| line7 | T2 | 0.20 | B1-QP-estim-yawlim | 0.253 [0.212, 0.293] | 67% | 0.198 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.10 | B1 | 0.390 [0.338, 0.443] | 69% | 0.409 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.10 | B1-QP | 0.448 [0.399, 0.493] | 78% | 0.471 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.10 | B3 INDI | 0.636 [0.634, 0.638] | 100% | — | 無舊值 |
| line7 | POS_STEP | 0.10 | B3+adapt | 0.631 [0.630, 0.633] | 100% | — | 無舊值 |
| line7 | POS_STEP | 0.10 | B1-QP-estim-yawlim | 0.640 [0.639, 0.641] | 100% | 0.639 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.15 | B1 | 0.221 [0.170, 0.277] | 42% | 0.208 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.15 | B1-QP | 0.297 [0.247, 0.348] | 62% | 0.295 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.15 | B3 INDI | 0.621 [0.615, 0.627] | 100% | — | 無舊值 |
| line7 | POS_STEP | 0.15 | B3+adapt | 0.601 [0.590, 0.611] | 100% | — | 無舊值 |
| line7 | POS_STEP | 0.15 | B1-QP-estim-yawlim | 0.636 [0.634, 0.638] | 100% | 0.635 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.20 | B1 | 0.140 [0.099, 0.183] | 30% | 0.125 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.20 | B1-QP | 0.199 [0.158, 0.243] | 49% | 0.196 | p3d_factorial_raw.json |
| line7 | POS_STEP | 0.20 | B3 INDI | 0.473 [0.424, 0.520] | 81% | — | 無舊值 |
| line7 | POS_STEP | 0.20 | B3+adapt | 0.466 [0.418, 0.509] | 81% | — | 無舊值 |
| line7 | POS_STEP | 0.20 | B1-QP-estim-yawlim | 0.512 [0.464, 0.556] | 85% | 0.532 | p3d_factorial_raw.json |
| line8 | T2 | 0.10 | B1 | 0.093 [0.064, 0.125] | 33% | 0.070 | p3d_factorial_raw.json |
| line8 | T2 | 0.10 | B1-QP | 0.213 [0.176, 0.249] | 66% | 0.137 | p3d_factorial_raw.json |
| line8 | T2 | 0.10 | B3 INDI | 0.262 [0.221, 0.302] | 72% | — | 無舊值 |
| line8 | T2 | 0.10 | B3+adapt | 0.255 [0.215, 0.296] | 73% | — | 無舊值 |
| line8 | T2 | 0.10 | B1-QP-estim-yawlim | 0.380 [0.350, 0.409] | 97% | 0.353 | p3d_factorial_raw.json |
| line8 | T2 | 0.15 | B1 | 0.024 [0.009, 0.044] | 9% | 0.029 | p3d_factorial_raw.json |
| line8 | T2 | 0.15 | B1-QP | 0.106 [0.075, 0.138] | 39% | 0.083 | p3d_factorial_raw.json |
| line8 | T2 | 0.15 | B3 INDI | 0.225 [0.184, 0.266] | 63% | — | 無舊值 |
| line8 | T2 | 0.15 | B3+adapt | 0.202 [0.160, 0.243] | 58% | — | 無舊值 |
| line8 | T2 | 0.15 | B1-QP-estim-yawlim | 0.335 [0.302, 0.370] | 88% | 0.301 | p3d_factorial_raw.json |
| line8 | T2 | 0.20 | B1 | 0.003 [0.000, 0.009] | 4% | 0.007 | p3d_factorial_raw.json |
| line8 | T2 | 0.20 | B1-QP | 0.034 [0.018, 0.053] | 17% | 0.040 | p3d_factorial_raw.json |
| line8 | T2 | 0.20 | B3 INDI | 0.163 [0.124, 0.202] | 51% | — | 無舊值 |
| line8 | T2 | 0.20 | B3+adapt | 0.113 [0.079, 0.151] | 36% | — | 無舊值 |
| line8 | T2 | 0.20 | B1-QP-estim-yawlim | 0.231 [0.192, 0.272] | 64% | 0.198 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.10 | B1 | 0.303 [0.249, 0.358] | 56% | 0.335 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.10 | B1-QP | 0.426 [0.374, 0.473] | 77% | 0.475 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.10 | B3 INDI | 0.618 [0.608, 0.625] | 100% | — | 無舊值 |
| line8 | POS_STEP | 0.10 | B3+adapt | 0.619 [0.616, 0.622] | 100% | — | 無舊值 |
| line8 | POS_STEP | 0.10 | B1-QP-estim-yawlim | 0.636 [0.634, 0.638] | 100% | 0.637 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.15 | B1 | 0.180 [0.132, 0.228] | 36% | 0.204 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.15 | B1-QP | 0.277 [0.225, 0.327] | 53% | 0.303 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.15 | B3 INDI | 0.565 [0.533, 0.593] | 94% | — | 無舊值 |
| line8 | POS_STEP | 0.15 | B3+adapt | 0.547 [0.515, 0.574] | 95% | — | 無舊值 |
| line8 | POS_STEP | 0.15 | B1-QP-estim-yawlim | 0.625 [0.621, 0.628] | 100% | 0.630 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.20 | B1 | 0.079 [0.047, 0.114] | 19% | 0.089 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.20 | B1-QP | 0.146 [0.105, 0.190] | 32% | 0.162 | p3d_factorial_raw.json |
| line8 | POS_STEP | 0.20 | B3 INDI | 0.390 [0.335, 0.444] | 68% | — | 無舊值 |
| line8 | POS_STEP | 0.20 | B3+adapt | 0.394 [0.341, 0.446] | 69% | — | 無舊值 |
| line8 | POS_STEP | 0.20 | B1-QP-estim-yawlim | 0.439 [0.385, 0.490] | 74% | 0.522 | p3d_factorial_raw.json |

### E2 成對對比（A - B；逐 episode 成對）

| 構型 | 任務 | 質量 | A | B | confirmatory 差 [95% CI] | A/B 比 | 探索 seed 差 | 方向 |
|---|---|---|---|---|---|---|---|---|
| line6 | T2 | 0.10 | B1-QP-estim-yawlim | B1-QP | +0.164 [+0.135, +0.196] | 1.690 | +0.158 | 一致 |
| line6 | T2 | 0.10 | B1-QP-estim-yawlim | B3 INDI | +0.085 [+0.062, +0.112] | 1.269 | — | n/a |
| line6 | T2 | 0.10 | B1-QP-estim-yawlim | B3+adapt | +0.093 [+0.068, +0.121] | 1.300 | — | n/a |
| line6 | T2 | 0.15 | B1-QP-estim-yawlim | B1-QP | +0.218 [+0.188, +0.250] | 2.417 | +0.257 | 一致 |
| line6 | T2 | 0.15 | B1-QP-estim-yawlim | B3 INDI | +0.100 [+0.074, +0.130] | 1.369 | — | n/a |
| line6 | T2 | 0.15 | B1-QP-estim-yawlim | B3+adapt | +0.099 [+0.072, +0.129] | 1.361 | — | n/a |
| line6 | T2 | 0.20 | B1-QP-estim-yawlim | B1-QP | +0.224 [+0.187, +0.261] | 4.949 | +0.200 | 一致 |
| line6 | T2 | 0.20 | B1-QP-estim-yawlim | B3 INDI | +0.059 [+0.040, +0.079] | 1.265 | — | n/a |
| line6 | T2 | 0.20 | B1-QP-estim-yawlim | B3+adapt | +0.071 [+0.051, +0.094] | 1.341 | — | n/a |
| line6 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B1-QP | +0.167 [+0.125, +0.214] | 1.353 | +0.140 | 一致 |
| line6 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B3 INDI | +0.002 [+0.002, +0.003] | 1.004 | — | n/a |
| line6 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B3+adapt | +0.004 [+0.004, +0.005] | 1.007 | — | n/a |
| line6 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B1-QP | +0.298 [+0.252, +0.347] | 1.876 | +0.284 | 一致 |
| line6 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B3 INDI | +0.007 [+0.006, +0.009] | 1.011 | — | n/a |
| line6 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B3+adapt | +0.014 [+0.010, +0.018] | 1.022 | — | n/a |
| line6 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B1-QP | +0.349 [+0.312, +0.389] | 2.424 | +0.347 | 一致 |
| line6 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B3 INDI | +0.073 [+0.049, +0.099] | 1.140 | — | n/a |
| line6 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B3+adapt | +0.070 [+0.046, +0.096] | 1.133 | — | n/a |
| line7 | T2 | 0.10 | B1-QP-estim-yawlim | B1-QP | +0.179 [+0.146, +0.213] | 1.852 | +0.201 | 一致 |
| line7 | T2 | 0.10 | B1-QP-estim-yawlim | B3 INDI | +0.107 [+0.078, +0.139] | 1.380 | — | n/a |
| line7 | T2 | 0.10 | B1-QP-estim-yawlim | B3+adapt | +0.098 [+0.073, +0.128] | 1.339 | — | n/a |
| line7 | T2 | 0.15 | B1-QP-estim-yawlim | B1-QP | +0.219 [+0.187, +0.251] | 2.651 | +0.230 | 一致 |
| line7 | T2 | 0.15 | B1-QP-estim-yawlim | B3 INDI | +0.101 [+0.075, +0.130] | 1.402 | — | n/a |
| line7 | T2 | 0.15 | B1-QP-estim-yawlim | B3+adapt | +0.114 [+0.084, +0.146] | 1.477 | — | n/a |
| line7 | T2 | 0.20 | B1-QP-estim-yawlim | B1-QP | +0.200 [+0.165, +0.234] | 4.771 | +0.167 | 一致 |
| line7 | T2 | 0.20 | B1-QP-estim-yawlim | B3 INDI | +0.080 [+0.054, +0.108] | 1.461 | — | n/a |
| line7 | T2 | 0.20 | B1-QP-estim-yawlim | B3+adapt | +0.097 [+0.071, +0.125] | 1.624 | — | n/a |
| line7 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B1-QP | +0.193 [+0.147, +0.242] | 1.431 | +0.168 | 一致 |
| line7 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B3 INDI | +0.005 [+0.003, +0.006] | 1.007 | — | n/a |
| line7 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B3+adapt | +0.009 [+0.008, +0.010] | 1.014 | — | n/a |
| line7 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B1-QP | +0.339 [+0.289, +0.389] | 2.142 | +0.339 | 一致 |
| line7 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B3 INDI | +0.015 [+0.010, +0.020] | 1.024 | — | n/a |
| line7 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B3+adapt | +0.035 [+0.025, +0.045] | 1.058 | — | n/a |
| line7 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B1-QP | +0.313 [+0.266, +0.358] | 2.570 | +0.336 | 一致 |
| line7 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B3 INDI | +0.038 [+0.025, +0.053] | 1.080 | — | n/a |
| line7 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B3+adapt | +0.046 [+0.027, +0.065] | 1.099 | — | n/a |
| line8 | T2 | 0.10 | B1-QP-estim-yawlim | B1-QP | +0.167 [+0.137, +0.198] | 1.786 | +0.216 | 一致 |
| line8 | T2 | 0.10 | B1-QP-estim-yawlim | B3 INDI | +0.119 [+0.089, +0.151] | 1.453 | — | n/a |
| line8 | T2 | 0.10 | B1-QP-estim-yawlim | B3+adapt | +0.125 [+0.097, +0.156] | 1.489 | — | n/a |
| line8 | T2 | 0.15 | B1-QP-estim-yawlim | B1-QP | +0.229 [+0.196, +0.263] | 3.158 | +0.218 | 一致 |
| line8 | T2 | 0.15 | B1-QP-estim-yawlim | B3 INDI | +0.111 [+0.081, +0.143] | 1.491 | — | n/a |
| line8 | T2 | 0.15 | B1-QP-estim-yawlim | B3+adapt | +0.134 [+0.103, +0.167] | 1.664 | — | n/a |
| line8 | T2 | 0.20 | B1-QP-estim-yawlim | B1-QP | +0.197 [+0.162, +0.233] | — | +0.158 | 一致 |
| line8 | T2 | 0.20 | B1-QP-estim-yawlim | B3 INDI | +0.068 [+0.038, +0.099] | 1.418 | — | n/a |
| line8 | T2 | 0.20 | B1-QP-estim-yawlim | B3+adapt | +0.118 [+0.090, +0.148] | 2.043 | — | n/a |
| line8 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B1-QP | +0.210 [+0.163, +0.262] | 1.494 | +0.162 | 一致 |
| line8 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B3 INDI | +0.018 [+0.012, +0.026] | 1.030 | — | n/a |
| line8 | POS_STEP | 0.10 | B1-QP-estim-yawlim | B3+adapt | +0.017 [+0.015, +0.019] | 1.028 | — | n/a |
| line8 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B1-QP | +0.347 [+0.299, +0.399] | 2.252 | +0.327 | 一致 |
| line8 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B3 INDI | +0.060 [+0.035, +0.089] | 1.105 | — | n/a |
| line8 | POS_STEP | 0.15 | B1-QP-estim-yawlim | B3+adapt | +0.078 [+0.052, +0.107] | 1.142 | — | n/a |
| line8 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B1-QP | +0.293 [+0.245, +0.341] | 3.007 | +0.360 | 一致 |
| line8 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B3 INDI | +0.049 [+0.032, +0.068] | 1.125 | — | n/a |
| line8 | POS_STEP | 0.20 | B1-QP-estim-yawlim | B3+adapt | +0.045 [-0.004, +0.092] | 1.115 | — | n/a |

## E3：offcenter（payload15 + p3d_factorial.OFFSET）

| 構型 | 任務 | 方法 | confirmatory 分數 [95% CI] | 成功率 | 探索 seed 分數 | 出處 |
|---|---|---|---|---|---|---|
| line6 | POS_STEP | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-estim-yawlim | 0.548 [0.538, 0.558] | 100% | 0.552 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-informed-offset-yawlim | 0.595 [0.589, 0.601] | 100% | 0.595 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-estim-yawlim | 0.210 [0.172, 0.249] | 67% | 0.197 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-informed-offset-yawlim | 0.209 [0.170, 0.247] | 66% | 0.204 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-estim-yawlim | 0.498 [0.481, 0.514] | 100% | 0.509 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-informed-offset-yawlim | 0.559 [0.549, 0.569] | 100% | 0.565 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-estim-yawlim | 0.182 [0.147, 0.219] | 59% | 0.158 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-informed-offset-yawlim | 0.181 [0.146, 0.219] | 57% | 0.161 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-estim-yawlim | 0.385 [0.346, 0.421] | 85% | 0.410 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-informed-offset-yawlim | 0.494 [0.477, 0.510] | 100% | 0.505 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-estim-yawlim | 0.163 [0.130, 0.199] | 53% | 0.131 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-informed-offset-yawlim | 0.168 [0.134, 0.204] | 56% | 0.136 | p3d_factorial_raw.json |

### E3 成對對比（A - B；逐 episode 成對）

| 構型 | 任務 | A | B | confirmatory 差 [95% CI] | A/B 比 | 探索 seed 差 | 方向 |
|---|---|---|---|---|---|---|---|
| line6 | POS_STEP | B1-QP-informed-offset-yawlim | B1-QP-estim-yawlim | +0.047 [+0.041, +0.052] | 1.086 | +0.043 | 一致 |
| line6 | POS_STEP | B1-QP-estim-yawlim | B1-QP-yawlim | +0.548 [+0.538, +0.558] | — | +0.552 | 一致 |
| line6 | T2 | B1-QP-informed-offset-yawlim | B1-QP-estim-yawlim | -0.001 [-0.008, +0.006] | 0.997 | +0.007 | 不一致 |
| line6 | T2 | B1-QP-estim-yawlim | B1-QP-yawlim | +0.210 [+0.172, +0.249] | — | +0.197 | 一致 |
| line7 | POS_STEP | B1-QP-informed-offset-yawlim | B1-QP-estim-yawlim | +0.061 [+0.052, +0.070] | 1.123 | +0.055 | 一致 |
| line7 | POS_STEP | B1-QP-estim-yawlim | B1-QP-yawlim | +0.498 [+0.481, +0.514] | — | +0.509 | 一致 |
| line7 | T2 | B1-QP-informed-offset-yawlim | B1-QP-estim-yawlim | -0.001 [-0.010, +0.007] | 0.994 | +0.003 | 不一致 |
| line7 | T2 | B1-QP-estim-yawlim | B1-QP-yawlim | +0.182 [+0.147, +0.219] | — | +0.158 | 一致 |
| line8 | POS_STEP | B1-QP-informed-offset-yawlim | B1-QP-estim-yawlim | +0.108 [+0.084, +0.134] | 1.281 | +0.095 | 一致 |
| line8 | POS_STEP | B1-QP-estim-yawlim | B1-QP-yawlim | +0.385 [+0.346, +0.421] | — | +0.410 | 一致 |
| line8 | T2 | B1-QP-informed-offset-yawlim | B1-QP-estim-yawlim | +0.005 [-0.007, +0.016] | 1.030 | +0.005 | 一致 |
| line8 | T2 | B1-QP-estim-yawlim | B1-QP-yawlim | +0.163 [+0.130, +0.199] | — | +0.131 | 一致 |

### E3 T2+POS_STEP 合併（逐 episode 平均 (T2+POS_STEP)/2；次要，不判定）

| 構型 | 方法 | confirmatory [95% CI] | 探索 seed 分數 |
|---|---|---|---|
| line6 | B1-QP | 0.000 [0.000, 0.000] | 0.000 |
| line6 | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0.000 |
| line6 | B1-QP-estim-yawlim | 0.379 [0.359, 0.399] | 0.375 |
| line6 | B1-QP-informed-offset-yawlim | 0.402 [0.382, 0.422] | 0.400 |
| line7 | B1-QP | 0.000 [0.000, 0.000] | 0.000 |
| line7 | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0.000 |
| line7 | B1-QP-estim-yawlim | 0.340 [0.320, 0.360] | 0.334 |
| line7 | B1-QP-informed-offset-yawlim | 0.370 [0.352, 0.389] | 0.363 |
| line8 | B1-QP | 0.000 [0.000, 0.000] | 0.000 |
| line8 | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0.000 |
| line8 | B1-QP-estim-yawlim | 0.274 [0.250, 0.297] | 0.270 |
| line8 | B1-QP-informed-offset-yawlim | 0.331 [0.312, 0.350] | 0.320 |

| 構型 | informed-offset - estim-yawlim（T2+POS） [95% CI] | 探索 seed 差 | 方向 |
|---|---|---|---|
| line6 | +0.023 [+0.019, +0.028] | +0.025 | 一致 |
| line7 | +0.030 [+0.024, +0.036] | +0.029 | 一致 |
| line8 | +0.057 [+0.043, +0.070] | +0.050 | 一致 |

### E3 一致性檢查（prereg 5.b 已填的 confirmatory 分數，容差 0.002）

| 構型 | 任務 | 方法 | 本次 | prereg 5.b | 差 | 判定 |
|---|---|---|---|---|---|---|
| line6 | POS_STEP | B1-QP-estim-yawlim | 0.548 | 0.548 | +0.0001 | 符合 |
| line6 | POS_STEP | B1-QP-informed-offset-yawlim | 0.595 | 0.595 | +0.0000 | 符合 |
| line7 | POS_STEP | B1-QP-estim-yawlim | 0.498 | 0.498 | -0.0005 | 符合 |
| line7 | POS_STEP | B1-QP-informed-offset-yawlim | 0.559 | 0.559 | -0.0002 | 符合 |
| line8 | POS_STEP | B1-QP-estim-yawlim | 0.385 | 0.385 | +0.0003 | 符合 |
| line8 | POS_STEP | B1-QP-informed-offset-yawlim | 0.494 | 0.494 | -0.0004 | 符合 |
| line6 | T2 | B1-QP-estim-yawlim | 0.210 | 0.210 | -0.0005 | 符合 |
| line6 | T2 | B1-QP-informed-offset-yawlim | 0.209 | 0.209 | -0.0001 | 符合 |
| line7 | T2 | B1-QP-estim-yawlim | 0.182 | 0.182 | +0.0004 | 符合 |
| line7 | T2 | B1-QP-informed-offset-yawlim | 0.181 | 0.181 | +0.0003 | 符合 |
| line8 | T2 | B1-QP-estim-yawlim | 0.163 | 0.163 | +0.0001 | 符合 |
| line8 | T2 | B1-QP-informed-offset-yawlim | 0.168 | 0.168 | +0.0000 | 符合 |

## NOM：nominal（無 payload）

| 構型 | 任務 | 方法 | confirmatory 分數 [95% CI] | 成功率 | 探索 seed 分數 | 出處 |
|---|---|---|---|---|---|---|
| line6 | T2 | B1 | 0.351 [0.319, 0.383] | 92% | 0.314 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP | 0.365 [0.335, 0.397] | 95% | 0.339 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-yawlim | 0.417 [0.393, 0.440] | 100% | 0.387 | p3d_factorial_raw.json |
| line6 | T2 | B1-QP-estim-yawlim | 0.417 [0.393, 0.440] | 100% | 0.387 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1 | 0.642 [0.641, 0.642] | 100% | 0.642 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP | 0.642 [0.641, 0.642] | 100% | 0.642 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-yawlim | 0.642 [0.641, 0.642] | 100% | 0.642 | p3d_factorial_raw.json |
| line6 | POS_STEP | B1-QP-estim-yawlim | 0.641 [0.641, 0.642] | 100% | 0.641 | p3d_factorial_raw.json |
| line7 | T2 | B1 | 0.286 [0.248, 0.325] | 78% | 0.266 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP | 0.341 [0.307, 0.376] | 89% | 0.305 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-yawlim | 0.415 [0.390, 0.438] | 100% | 0.383 | p3d_factorial_raw.json |
| line7 | T2 | B1-QP-estim-yawlim | 0.415 [0.390, 0.438] | 100% | 0.383 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1 | 0.642 [0.641, 0.643] | 100% | 0.641 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP | 0.642 [0.641, 0.643] | 100% | 0.641 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-yawlim | 0.642 [0.641, 0.642] | 100% | 0.641 | p3d_factorial_raw.json |
| line7 | POS_STEP | B1-QP-estim-yawlim | 0.642 [0.641, 0.642] | 100% | 0.641 | p3d_factorial_raw.json |
| line8 | T2 | B1 | 0.230 [0.187, 0.274] | 62% | 0.211 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP | 0.328 [0.293, 0.363] | 87% | 0.274 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-yawlim | 0.411 [0.385, 0.434] | 100% | 0.378 | p3d_factorial_raw.json |
| line8 | T2 | B1-QP-estim-yawlim | 0.411 [0.385, 0.435] | 100% | 0.378 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1 | 0.634 [0.632, 0.636] | 100% | 0.635 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP | 0.640 [0.639, 0.641] | 100% | 0.640 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-yawlim | 0.640 [0.639, 0.641] | 100% | 0.640 | p3d_factorial_raw.json |
| line8 | POS_STEP | B1-QP-estim-yawlim | 0.640 [0.639, 0.641] | 100% | 0.640 | p3d_factorial_raw.json |

### 6.1 退化比（payload15 分數 / 標稱分數，同構型同任務同 seed）

| 構型 | 任務 | 方法 | confirmatory payload15 | confirmatory 標稱 | 退化比 | 探索 seed 退化比 |
|---|---|---|---|---|---|---|
| line6 | T2 | B1 | 0.000 | 0.351 | 0.000 | 0.000 |
| line6 | T2 | B1-QP | 0.011 | 0.365 | 0.031 | 0.006 |
| line6 | T2 | B1-QP-yawlim | 0.021 | 0.417 | 0.049 | 0.030 |
| line6 | T2 | B1-QP-estim-yawlim | 0.340 | 0.417 | 0.815 | 0.794 |
| line6 | POS_STEP | B1 | 0.000 | 0.642 | 0.000 | 0.000 |
| line6 | POS_STEP | B1-QP | 0.104 | 0.642 | 0.161 | 0.185 |
| line6 | POS_STEP | B1-QP-yawlim | 0.107 | 0.642 | 0.167 | 0.189 |
| line6 | POS_STEP | B1-QP-estim-yawlim | 0.633 | 0.641 | 0.987 | 0.986 |
| line7 | T2 | B1 | 0.000 | 0.286 | 0.000 | 0.000 |
| line7 | T2 | B1-QP | 0.010 | 0.341 | 0.029 | 0.005 |
| line7 | T2 | B1-QP-yawlim | 0.011 | 0.415 | 0.025 | 0.018 |
| line7 | T2 | B1-QP-estim-yawlim | 0.310 | 0.415 | 0.748 | 0.755 |
| line7 | POS_STEP | B1 | 0.000 | 0.642 | 0.000 | 0.000 |
| line7 | POS_STEP | B1-QP | 0.000 | 0.642 | 0.000 | 0.000 |
| line7 | POS_STEP | B1-QP-yawlim | 0.000 | 0.642 | 0.000 | 0.000 |
| line7 | POS_STEP | B1-QP-estim-yawlim | 0.628 | 0.642 | 0.978 | 0.976 |
| line8 | T2 | B1 | 0.000 | 0.230 | 0.000 | 0.000 |
| line8 | T2 | B1-QP | 0.000 | 0.328 | 0.001 | 0.000 |
| line8 | T2 | B1-QP-yawlim | 0.009 | 0.411 | 0.023 | 0.001 |
| line8 | T2 | B1-QP-estim-yawlim | 0.283 | 0.411 | 0.689 | 0.674 |
| line8 | POS_STEP | B1 | 0.000 | 0.634 | 0.000 | 0.000 |
| line8 | POS_STEP | B1-QP | 0.000 | 0.640 | 0.000 | 0.000 |
| line8 | POS_STEP | B1-QP-yawlim | 0.000 | 0.640 | 0.000 | 0.000 |
| line8 | POS_STEP | B1-QP-estim-yawlim | 0.610 | 0.640 | 0.953 | 0.958 |

## EV：中途 payload 事件（rqd_classical）

| 構型 | 任務 | 條件 | 方法 | confirmatory 分數 [95% CI] | 成功率 | 探索 seed 分數 | 出處 |
|---|---|---|---|---|---|---|---|
| line6 | T2 | attach-center | B1-QP-yawlim | 0.234 [0.196, 0.272] | 68% | 0.227 | rqd_classical_raw.json |
| line6 | T2 | attach-center | estim-yawlim | 0.362 [0.330, 0.392] | 95% | 0.349 | rqd_classical_raw.json |
| line6 | T2 | attach-center | estim-reset-yawlim | 0.388 [0.358, 0.417] | 96% | 0.367 | rqd_classical_raw.json |
| line6 | T2 | attach-center | informed-event-yawlim | 0.389 [0.358, 0.418] | 96% | 0.364 | rqd_classical_raw.json |
| line6 | T2 | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line6 | T2 | attach-offcenter | estim-yawlim | 0.222 [0.186, 0.261] | 77% | 0.210 | rqd_classical_raw.json |
| line6 | T2 | attach-offcenter | estim-reset-yawlim | 0.366 [0.332, 0.396] | 92% | 0.339 | rqd_classical_raw.json |
| line6 | T2 | attach-offcenter | informed-event-yawlim | 0.371 [0.338, 0.400] | 92% | 0.342 | rqd_classical_raw.json |
| line6 | T2 | drop-center | B1-QP-yawlim | 0.203 [0.165, 0.242] | 59% | 0.201 | rqd_classical_raw.json |
| line6 | T2 | drop-center | estim-yawlim | 0.365 [0.332, 0.396] | 93% | 0.354 | rqd_classical_raw.json |
| line6 | T2 | drop-center | estim-reset-yawlim | 0.388 [0.359, 0.416] | 96% | 0.373 | rqd_classical_raw.json |
| line6 | T2 | drop-center | informed-event-yawlim | 0.389 [0.360, 0.417] | 96% | 0.371 | rqd_classical_raw.json |
| line6 | T2 | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line6 | T2 | drop-offcenter | estim-yawlim | 0.250 [0.215, 0.286] | 78% | 0.222 | rqd_classical_raw.json |
| line6 | T2 | drop-offcenter | estim-reset-yawlim | 0.360 [0.328, 0.391] | 92% | 0.340 | rqd_classical_raw.json |
| line6 | T2 | drop-offcenter | informed-event-yawlim | 0.367 [0.337, 0.396] | 94% | 0.347 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-center | B1-QP-yawlim | 0.499 [0.464, 0.532] | 98% | 0.503 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-center | estim-yawlim | 0.631 [0.630, 0.632] | 100% | 0.631 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-center | estim-reset-yawlim | 0.641 [0.641, 0.642] | 100% | 0.641 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-center | informed-event-yawlim | 0.642 [0.641, 0.642] | 100% | 0.641 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-offcenter | estim-yawlim | 0.541 [0.535, 0.546] | 100% | 0.539 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.640 [0.639, 0.640] | 100% | 0.639 | rqd_classical_raw.json |
| line6 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.641 [0.640, 0.642] | 100% | 0.641 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-center | B1-QP-yawlim | 0.376 [0.323, 0.430] | 70% | 0.375 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-center | estim-yawlim | 0.629 [0.627, 0.630] | 100% | 0.628 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-center | estim-reset-yawlim | 0.639 [0.637, 0.640] | 100% | 0.638 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-center | informed-event-yawlim | 0.640 [0.639, 0.641] | 100% | 0.639 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-offcenter | estim-yawlim | 0.554 [0.548, 0.559] | 100% | 0.552 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.617 [0.610, 0.624] | 100% | 0.614 | rqd_classical_raw.json |
| line6 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.629 [0.624, 0.633] | 100% | 0.625 | rqd_classical_raw.json |
| line6 | HOVER | attach-center | B1-QP-yawlim | 0.700 [0.660, 0.740] | 100% | 0.688 | rqd_classical_raw.json |
| line6 | HOVER | attach-center | estim-yawlim | 0.915 [0.914, 0.917] | 100% | 0.915 | rqd_classical_raw.json |
| line6 | HOVER | attach-center | estim-reset-yawlim | 0.987 [0.986, 0.988] | 100% | 0.987 | rqd_classical_raw.json |
| line6 | HOVER | attach-center | informed-event-yawlim | 0.990 [0.989, 0.990] | 100% | 0.989 | rqd_classical_raw.json |
| line6 | HOVER | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line6 | HOVER | attach-offcenter | estim-yawlim | 0.724 [0.715, 0.731] | 100% | 0.723 | rqd_classical_raw.json |
| line6 | HOVER | attach-offcenter | estim-reset-yawlim | 0.962 [0.961, 0.962] | 100% | 0.962 | rqd_classical_raw.json |
| line6 | HOVER | attach-offcenter | informed-event-yawlim | 0.975 [0.975, 0.975] | 100% | 0.975 | rqd_classical_raw.json |
| line6 | HOVER | drop-center | B1-QP-yawlim | 0.602 [0.535, 0.670] | 84% | 0.592 | rqd_classical_raw.json |
| line6 | HOVER | drop-center | estim-yawlim | 0.913 [0.912, 0.915] | 100% | 0.913 | rqd_classical_raw.json |
| line6 | HOVER | drop-center | estim-reset-yawlim | 0.985 [0.985, 0.986] | 100% | 0.985 | rqd_classical_raw.json |
| line6 | HOVER | drop-center | informed-event-yawlim | 0.988 [0.987, 0.988] | 100% | 0.987 | rqd_classical_raw.json |
| line6 | HOVER | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line6 | HOVER | drop-offcenter | estim-yawlim | 0.771 [0.767, 0.774] | 100% | 0.773 | rqd_classical_raw.json |
| line6 | HOVER | drop-offcenter | estim-reset-yawlim | 0.948 [0.948, 0.948] | 100% | 0.948 | rqd_classical_raw.json |
| line6 | HOVER | drop-offcenter | informed-event-yawlim | 0.957 [0.957, 0.957] | 100% | 0.957 | rqd_classical_raw.json |
| line7 | T2 | attach-center | B1-QP-yawlim | 0.219 [0.180, 0.257] | 62% | 0.211 | rqd_classical_raw.json |
| line7 | T2 | attach-center | estim-yawlim | 0.346 [0.312, 0.379] | 91% | 0.332 | rqd_classical_raw.json |
| line7 | T2 | attach-center | estim-reset-yawlim | 0.383 [0.352, 0.413] | 96% | 0.352 | rqd_classical_raw.json |
| line7 | T2 | attach-center | informed-event-yawlim | 0.384 [0.353, 0.414] | 96% | 0.352 | rqd_classical_raw.json |
| line7 | T2 | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line7 | T2 | attach-offcenter | estim-yawlim | 0.216 [0.180, 0.254] | 74% | 0.183 | rqd_classical_raw.json |
| line7 | T2 | attach-offcenter | estim-reset-yawlim | 0.346 [0.310, 0.380] | 88% | 0.315 | rqd_classical_raw.json |
| line7 | T2 | attach-offcenter | informed-event-yawlim | 0.357 [0.322, 0.388] | 90% | 0.324 | rqd_classical_raw.json |
| line7 | T2 | drop-center | B1-QP-yawlim | 0.207 [0.168, 0.244] | 62% | 0.196 | rqd_classical_raw.json |
| line7 | T2 | drop-center | estim-yawlim | 0.362 [0.330, 0.393] | 92% | 0.341 | rqd_classical_raw.json |
| line7 | T2 | drop-center | estim-reset-yawlim | 0.387 [0.357, 0.413] | 96% | 0.361 | rqd_classical_raw.json |
| line7 | T2 | drop-center | informed-event-yawlim | 0.388 [0.359, 0.415] | 97% | 0.360 | rqd_classical_raw.json |
| line7 | T2 | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line7 | T2 | drop-offcenter | estim-yawlim | 0.242 [0.208, 0.277] | 80% | 0.203 | rqd_classical_raw.json |
| line7 | T2 | drop-offcenter | estim-reset-yawlim | 0.356 [0.325, 0.386] | 93% | 0.316 | rqd_classical_raw.json |
| line7 | T2 | drop-offcenter | informed-event-yawlim | 0.358 [0.325, 0.388] | 92% | 0.330 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-center | B1-QP-yawlim | 0.476 [0.430, 0.519] | 82% | 0.468 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-center | estim-yawlim | 0.631 [0.630, 0.632] | 100% | 0.631 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-center | estim-reset-yawlim | 0.641 [0.640, 0.642] | 100% | 0.641 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-center | informed-event-yawlim | 0.641 [0.640, 0.642] | 100% | 0.641 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-offcenter | estim-yawlim | 0.539 [0.533, 0.545] | 100% | 0.537 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.639 [0.638, 0.640] | 100% | 0.639 | rqd_classical_raw.json |
| line7 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.640 [0.640, 0.641] | 100% | 0.640 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-center | B1-QP-yawlim | 0.398 [0.348, 0.446] | 75% | 0.372 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-center | estim-yawlim | 0.625 [0.623, 0.627] | 100% | 0.625 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-center | estim-reset-yawlim | 0.635 [0.633, 0.637] | 100% | 0.635 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-center | informed-event-yawlim | 0.636 [0.635, 0.638] | 100% | 0.636 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-offcenter | estim-yawlim | 0.539 [0.530, 0.547] | 100% | 0.538 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.604 [0.594, 0.613] | 100% | 0.601 | rqd_classical_raw.json |
| line7 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.618 [0.613, 0.624] | 100% | 0.617 | rqd_classical_raw.json |
| line7 | HOVER | attach-center | B1-QP-yawlim | 0.644 [0.578, 0.708] | 79% | 0.621 | rqd_classical_raw.json |
| line7 | HOVER | attach-center | estim-yawlim | 0.916 [0.915, 0.918] | 100% | 0.916 | rqd_classical_raw.json |
| line7 | HOVER | attach-center | estim-reset-yawlim | 0.987 [0.986, 0.987] | 100% | 0.986 | rqd_classical_raw.json |
| line7 | HOVER | attach-center | informed-event-yawlim | 0.989 [0.988, 0.990] | 100% | 0.988 | rqd_classical_raw.json |
| line7 | HOVER | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line7 | HOVER | attach-offcenter | estim-yawlim | 0.723 [0.714, 0.732] | 100% | 0.723 | rqd_classical_raw.json |
| line7 | HOVER | attach-offcenter | estim-reset-yawlim | 0.961 [0.961, 0.962] | 100% | 0.961 | rqd_classical_raw.json |
| line7 | HOVER | attach-offcenter | informed-event-yawlim | 0.975 [0.975, 0.975] | 100% | 0.975 | rqd_classical_raw.json |
| line7 | HOVER | drop-center | B1-QP-yawlim | 0.623 [0.555, 0.689] | 76% | 0.598 | rqd_classical_raw.json |
| line7 | HOVER | drop-center | estim-yawlim | 0.914 [0.913, 0.915] | 100% | 0.914 | rqd_classical_raw.json |
| line7 | HOVER | drop-center | estim-reset-yawlim | 0.985 [0.984, 0.986] | 100% | 0.984 | rqd_classical_raw.json |
| line7 | HOVER | drop-center | informed-event-yawlim | 0.987 [0.986, 0.988] | 100% | 0.987 | rqd_classical_raw.json |
| line7 | HOVER | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line7 | HOVER | drop-offcenter | estim-yawlim | 0.772 [0.768, 0.776] | 100% | 0.776 | rqd_classical_raw.json |
| line7 | HOVER | drop-offcenter | estim-reset-yawlim | 0.948 [0.947, 0.948] | 100% | 0.948 | rqd_classical_raw.json |
| line7 | HOVER | drop-offcenter | informed-event-yawlim | 0.957 [0.957, 0.957] | 100% | 0.957 | rqd_classical_raw.json |
| line8 | T2 | attach-center | B1-QP-yawlim | 0.187 [0.149, 0.226] | 55% | 0.198 | rqd_classical_raw.json |
| line8 | T2 | attach-center | estim-yawlim | 0.317 [0.283, 0.353] | 85% | 0.323 | rqd_classical_raw.json |
| line8 | T2 | attach-center | estim-reset-yawlim | 0.368 [0.336, 0.400] | 94% | 0.341 | rqd_classical_raw.json |
| line8 | T2 | attach-center | informed-event-yawlim | 0.370 [0.337, 0.401] | 94% | 0.342 | rqd_classical_raw.json |
| line8 | T2 | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line8 | T2 | attach-offcenter | estim-yawlim | 0.195 [0.158, 0.235] | 66% | 0.173 | rqd_classical_raw.json |
| line8 | T2 | attach-offcenter | estim-reset-yawlim | 0.319 [0.281, 0.356] | 83% | 0.302 | rqd_classical_raw.json |
| line8 | T2 | attach-offcenter | informed-event-yawlim | 0.327 [0.290, 0.363] | 86% | 0.312 | rqd_classical_raw.json |
| line8 | T2 | drop-center | B1-QP-yawlim | 0.172 [0.135, 0.209] | 53% | 0.188 | rqd_classical_raw.json |
| line8 | T2 | drop-center | estim-yawlim | 0.351 [0.318, 0.383] | 91% | 0.329 | rqd_classical_raw.json |
| line8 | T2 | drop-center | estim-reset-yawlim | 0.375 [0.344, 0.403] | 95% | 0.349 | rqd_classical_raw.json |
| line8 | T2 | drop-center | informed-event-yawlim | 0.378 [0.348, 0.405] | 96% | 0.350 | rqd_classical_raw.json |
| line8 | T2 | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line8 | T2 | drop-offcenter | estim-yawlim | 0.223 [0.189, 0.259] | 76% | 0.182 | rqd_classical_raw.json |
| line8 | T2 | drop-offcenter | estim-reset-yawlim | 0.335 [0.300, 0.368] | 90% | 0.303 | rqd_classical_raw.json |
| line8 | T2 | drop-offcenter | informed-event-yawlim | 0.342 [0.309, 0.375] | 90% | 0.311 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-center | B1-QP-yawlim | 0.467 [0.418, 0.513] | 81% | 0.462 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-center | estim-yawlim | 0.621 [0.607, 0.629] | 99% | 0.630 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-center | estim-reset-yawlim | 0.638 [0.637, 0.639] | 100% | 0.640 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-center | informed-event-yawlim | 0.639 [0.638, 0.640] | 100% | 0.640 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-offcenter | estim-yawlim | 0.519 [0.499, 0.535] | 98% | 0.523 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.623 [0.603, 0.636] | 98% | 0.636 | rqd_classical_raw.json |
| line8 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.633 [0.625, 0.638] | 100% | 0.638 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-center | B1-QP-yawlim | 0.361 [0.309, 0.417] | 73% | 0.349 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-center | estim-yawlim | 0.618 [0.614, 0.621] | 100% | 0.621 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-center | estim-reset-yawlim | 0.627 [0.623, 0.630] | 100% | 0.630 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-center | informed-event-yawlim | 0.629 [0.626, 0.632] | 100% | 0.631 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-offcenter | estim-yawlim | 0.511 [0.495, 0.524] | 99% | 0.518 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.576 [0.559, 0.590] | 99% | 0.580 | rqd_classical_raw.json |
| line8 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.598 [0.589, 0.607] | 100% | 0.602 | rqd_classical_raw.json |
| line8 | HOVER | attach-center | B1-QP-yawlim | 0.657 [0.591, 0.719] | 80% | 0.665 | rqd_classical_raw.json |
| line8 | HOVER | attach-center | estim-yawlim | 0.917 [0.916, 0.918] | 100% | 0.917 | rqd_classical_raw.json |
| line8 | HOVER | attach-center | estim-reset-yawlim | 0.986 [0.985, 0.987] | 100% | 0.986 | rqd_classical_raw.json |
| line8 | HOVER | attach-center | informed-event-yawlim | 0.988 [0.988, 0.989] | 100% | 0.988 | rqd_classical_raw.json |
| line8 | HOVER | attach-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line8 | HOVER | attach-offcenter | estim-yawlim | 0.723 [0.714, 0.731] | 100% | 0.725 | rqd_classical_raw.json |
| line8 | HOVER | attach-offcenter | estim-reset-yawlim | 0.961 [0.960, 0.962] | 100% | 0.961 | rqd_classical_raw.json |
| line8 | HOVER | attach-offcenter | informed-event-yawlim | 0.975 [0.974, 0.975] | 100% | 0.975 | rqd_classical_raw.json |
| line8 | HOVER | drop-center | B1-QP-yawlim | 0.641 [0.574, 0.705] | 78% | 0.646 | rqd_classical_raw.json |
| line8 | HOVER | drop-center | estim-yawlim | 0.915 [0.914, 0.916] | 100% | 0.915 | rqd_classical_raw.json |
| line8 | HOVER | drop-center | estim-reset-yawlim | 0.985 [0.984, 0.985] | 100% | 0.984 | rqd_classical_raw.json |
| line8 | HOVER | drop-center | informed-event-yawlim | 0.987 [0.986, 0.988] | 100% | 0.986 | rqd_classical_raw.json |
| line8 | HOVER | drop-offcenter | B1-QP-yawlim | 0.000 [0.000, 0.000] | 0% | 0.000 | rqd_classical_raw.json |
| line8 | HOVER | drop-offcenter | estim-yawlim | 0.773 [0.769, 0.777] | 100% | 0.776 | rqd_classical_raw.json |
| line8 | HOVER | drop-offcenter | estim-reset-yawlim | 0.947 [0.947, 0.948] | 100% | 0.948 | rqd_classical_raw.json |
| line8 | HOVER | drop-offcenter | informed-event-yawlim | 0.957 [0.957, 0.957] | 100% | 0.957 | rqd_classical_raw.json |
| grid10 | T2 | attach-center | B1-QP-yawlim | 0.122 [0.091, 0.157] | 45% | 0.142 | rqd_classical_raw.json |
| grid10 | T2 | attach-center | estim-yawlim | 0.349 [0.315, 0.381] | 93% | 0.343 | rqd_classical_raw.json |
| grid10 | T2 | attach-center | estim-reset-yawlim | 0.394 [0.366, 0.421] | 98% | 0.371 | rqd_classical_raw.json |
| grid10 | T2 | attach-center | informed-event-yawlim | 0.395 [0.367, 0.422] | 98% | 0.372 | rqd_classical_raw.json |
| grid10 | T2 | attach-offcenter | B1-QP-yawlim | 0.107 [0.074, 0.144] | 34% | 0.101 | rqd_classical_raw.json |
| grid10 | T2 | attach-offcenter | estim-yawlim | 0.328 [0.291, 0.363] | 88% | 0.318 | rqd_classical_raw.json |
| grid10 | T2 | attach-offcenter | estim-reset-yawlim | 0.378 [0.346, 0.409] | 94% | 0.365 | rqd_classical_raw.json |
| grid10 | T2 | attach-offcenter | informed-event-yawlim | 0.379 [0.347, 0.410] | 94% | 0.364 | rqd_classical_raw.json |
| grid10 | T2 | drop-center | B1-QP-yawlim | 0.116 [0.086, 0.153] | 46% | 0.138 | rqd_classical_raw.json |
| grid10 | T2 | drop-center | estim-yawlim | 0.369 [0.339, 0.398] | 97% | 0.344 | rqd_classical_raw.json |
| grid10 | T2 | drop-center | estim-reset-yawlim | 0.388 [0.360, 0.416] | 98% | 0.373 | rqd_classical_raw.json |
| grid10 | T2 | drop-center | informed-event-yawlim | 0.387 [0.358, 0.415] | 97% | 0.374 | rqd_classical_raw.json |
| grid10 | T2 | drop-offcenter | B1-QP-yawlim | 0.094 [0.065, 0.126] | 31% | 0.102 | rqd_classical_raw.json |
| grid10 | T2 | drop-offcenter | estim-yawlim | 0.359 [0.328, 0.388] | 95% | 0.332 | rqd_classical_raw.json |
| grid10 | T2 | drop-offcenter | estim-reset-yawlim | 0.382 [0.351, 0.409] | 96% | 0.363 | rqd_classical_raw.json |
| grid10 | T2 | drop-offcenter | informed-event-yawlim | 0.384 [0.354, 0.411] | 96% | 0.365 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-center | B1-QP-yawlim | 0.356 [0.316, 0.397] | 81% | 0.353 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-center | estim-yawlim | 0.623 [0.623, 0.624] | 100% | 0.623 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-center | estim-reset-yawlim | 0.642 [0.641, 0.642] | 100% | 0.641 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-center | informed-event-yawlim | 0.642 [0.641, 0.643] | 100% | 0.642 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-offcenter | B1-QP-yawlim | 0.251 [0.196, 0.305] | 44% | 0.220 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-offcenter | estim-yawlim | 0.618 [0.615, 0.621] | 100% | 0.617 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.641 [0.641, 0.642] | 100% | 0.641 | rqd_classical_raw.json |
| grid10 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.642 [0.641, 0.643] | 100% | 0.642 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-center | B1-QP-yawlim | 0.215 [0.172, 0.256] | 57% | 0.176 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-center | estim-yawlim | 0.619 [0.617, 0.621] | 100% | 0.620 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-center | estim-reset-yawlim | 0.636 [0.634, 0.638] | 100% | 0.637 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-center | informed-event-yawlim | 0.638 [0.637, 0.640] | 100% | 0.639 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-offcenter | B1-QP-yawlim | 0.195 [0.147, 0.242] | 45% | 0.180 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-offcenter | estim-yawlim | 0.613 [0.608, 0.618] | 100% | 0.610 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.631 [0.626, 0.635] | 100% | 0.629 | rqd_classical_raw.json |
| grid10 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.636 [0.634, 0.639] | 100% | 0.635 | rqd_classical_raw.json |
| grid10 | HOVER | attach-center | B1-QP-yawlim | 0.475 [0.422, 0.526] | 80% | 0.472 | rqd_classical_raw.json |
| grid10 | HOVER | attach-center | estim-yawlim | 0.884 [0.883, 0.886] | 100% | 0.884 | rqd_classical_raw.json |
| grid10 | HOVER | attach-center | estim-reset-yawlim | 0.980 [0.980, 0.981] | 100% | 0.980 | rqd_classical_raw.json |
| grid10 | HOVER | attach-center | informed-event-yawlim | 0.985 [0.985, 0.986] | 100% | 0.985 | rqd_classical_raw.json |
| grid10 | HOVER | attach-offcenter | B1-QP-yawlim | 0.316 [0.244, 0.393] | 42% | 0.332 | rqd_classical_raw.json |
| grid10 | HOVER | attach-offcenter | estim-yawlim | 0.872 [0.864, 0.880] | 100% | 0.874 | rqd_classical_raw.json |
| grid10 | HOVER | attach-offcenter | estim-reset-yawlim | 0.978 [0.976, 0.980] | 100% | 0.978 | rqd_classical_raw.json |
| grid10 | HOVER | attach-offcenter | informed-event-yawlim | 0.984 [0.984, 0.985] | 100% | 0.984 | rqd_classical_raw.json |
| grid10 | HOVER | drop-center | B1-QP-yawlim | 0.424 [0.368, 0.480] | 76% | 0.419 | rqd_classical_raw.json |
| grid10 | HOVER | drop-center | estim-yawlim | 0.887 [0.886, 0.888] | 100% | 0.887 | rqd_classical_raw.json |
| grid10 | HOVER | drop-center | estim-reset-yawlim | 0.975 [0.975, 0.976] | 100% | 0.975 | rqd_classical_raw.json |
| grid10 | HOVER | drop-center | informed-event-yawlim | 0.978 [0.978, 0.979] | 100% | 0.978 | rqd_classical_raw.json |
| grid10 | HOVER | drop-offcenter | B1-QP-yawlim | 0.303 [0.231, 0.378] | 42% | 0.321 | rqd_classical_raw.json |
| grid10 | HOVER | drop-offcenter | estim-yawlim | 0.881 [0.877, 0.886] | 100% | 0.883 | rqd_classical_raw.json |
| grid10 | HOVER | drop-offcenter | estim-reset-yawlim | 0.972 [0.970, 0.974] | 100% | 0.973 | rqd_classical_raw.json |
| grid10 | HOVER | drop-offcenter | informed-event-yawlim | 0.977 [0.976, 0.979] | 100% | 0.977 | rqd_classical_raw.json |

### EV 成對對比（A - B；逐 episode 成對）

| 構型 | 任務 | 條件 | A | B | confirmatory 差 [95% CI] | A/B 比 | 探索 seed 差 | 方向 |
|---|---|---|---|---|---|---|---|---|
| line6 | T2 | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.000, +0.001] | 1.001 | -0.003 | 不一致 |
| line6 | T2 | attach-center | estim-reset-yawlim | estim-yawlim | +0.026 [+0.017, +0.038] | 1.073 | +0.018 | 一致 |
| line6 | T2 | attach-center | estim-yawlim | B1-QP-yawlim | +0.128 [+0.098, +0.159] | 1.546 | +0.121 | 一致 |
| line6 | T2 | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.005 [+0.003, +0.007] | 1.013 | +0.003 | 一致 |
| line6 | T2 | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.144 [+0.117, +0.171] | 1.646 | +0.130 | 一致 |
| line6 | T2 | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.222 [+0.186, +0.261] | — | +0.210 | 一致 |
| line6 | T2 | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.000, +0.001] | 1.002 | -0.002 | 一致 |
| line6 | T2 | drop-center | estim-reset-yawlim | estim-yawlim | +0.023 [+0.010, +0.040] | 1.064 | +0.019 | 一致 |
| line6 | T2 | drop-center | estim-yawlim | B1-QP-yawlim | +0.162 [+0.128, +0.198] | 1.797 | +0.153 | 一致 |
| line6 | T2 | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.007 [+0.002, +0.013] | 1.018 | +0.008 | 一致 |
| line6 | T2 | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.110 [+0.086, +0.136] | 1.439 | +0.118 | 一致 |
| line6 | T2 | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.250 [+0.215, +0.286] | — | +0.222 | 一致 |
| line6 | POS_STEP | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.000 [+0.000, +0.000] | 1.000 | +0.000 | 一致 |
| line6 | POS_STEP | attach-center | estim-reset-yawlim | estim-yawlim | +0.010 [+0.010, +0.010] | 1.016 | +0.010 | 一致 |
| line6 | POS_STEP | attach-center | estim-yawlim | B1-QP-yawlim | +0.132 [+0.099, +0.167] | 1.265 | +0.128 | 一致 |
| line6 | POS_STEP | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.001, +0.002] | 1.002 | +0.001 | 一致 |
| line6 | POS_STEP | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.098 [+0.093, +0.104] | 1.182 | +0.100 | 一致 |
| line6 | POS_STEP | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.541 [+0.535, +0.546] | — | +0.539 | 一致 |
| line6 | POS_STEP | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.001, +0.002] | 1.002 | +0.001 | 一致 |
| line6 | POS_STEP | drop-center | estim-reset-yawlim | estim-yawlim | +0.010 [+0.010, +0.011] | 1.016 | +0.010 | 一致 |
| line6 | POS_STEP | drop-center | estim-yawlim | B1-QP-yawlim | +0.253 [+0.199, +0.305] | 1.671 | +0.253 | 一致 |
| line6 | POS_STEP | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.011 [+0.008, +0.015] | 1.018 | +0.011 | 一致 |
| line6 | POS_STEP | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.064 [+0.061, +0.067] | 1.115 | +0.063 | 一致 |
| line6 | POS_STEP | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.554 [+0.548, +0.559] | — | +0.552 | 一致 |
| line6 | HOVER | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.003] | 1.002 | +0.002 | 一致 |
| line6 | HOVER | attach-center | estim-reset-yawlim | estim-yawlim | +0.072 [+0.071, +0.073] | 1.078 | +0.072 | 一致 |
| line6 | HOVER | attach-center | estim-yawlim | B1-QP-yawlim | +0.215 [+0.177, +0.255] | 1.307 | +0.227 | 一致 |
| line6 | HOVER | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.014 [+0.013, +0.014] | 1.014 | +0.014 | 一致 |
| line6 | HOVER | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.238 [+0.231, +0.246] | 1.329 | +0.239 | 一致 |
| line6 | HOVER | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.724 [+0.715, +0.731] | — | +0.723 | 一致 |
| line6 | HOVER | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.003] | 1.002 | +0.002 | 一致 |
| line6 | HOVER | drop-center | estim-reset-yawlim | estim-yawlim | +0.072 [+0.071, +0.073] | 1.079 | +0.072 | 一致 |
| line6 | HOVER | drop-center | estim-yawlim | B1-QP-yawlim | +0.311 [+0.245, +0.378] | 1.517 | +0.321 | 一致 |
| line6 | HOVER | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.009 [+0.008, +0.009] | 1.009 | +0.009 | 一致 |
| line6 | HOVER | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.177 [+0.173, +0.181] | 1.230 | +0.175 | 一致 |
| line6 | HOVER | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.771 [+0.767, +0.774] | — | +0.773 | 一致 |
| line7 | T2 | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.001, +0.001] | 1.002 | +0.001 | 一致 |
| line7 | T2 | attach-center | estim-reset-yawlim | estim-yawlim | +0.038 [+0.021, +0.057] | 1.109 | +0.020 | 一致 |
| line7 | T2 | attach-center | estim-yawlim | B1-QP-yawlim | +0.127 [+0.096, +0.160] | 1.581 | +0.121 | 一致 |
| line7 | T2 | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.011 [+0.004, +0.019] | 1.031 | +0.008 | 一致 |
| line7 | T2 | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.129 [+0.104, +0.158] | 1.598 | +0.133 | 一致 |
| line7 | T2 | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.216 [+0.180, +0.254] | — | +0.183 | 一致 |
| line7 | T2 | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.000, +0.002] | 1.003 | -0.002 | 一致 |
| line7 | T2 | drop-center | estim-reset-yawlim | estim-yawlim | +0.024 [+0.010, +0.041] | 1.067 | +0.020 | 一致 |
| line7 | T2 | drop-center | estim-yawlim | B1-QP-yawlim | +0.155 [+0.122, +0.190] | 1.750 | +0.145 | 一致 |
| line7 | T2 | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.002 [-0.004, +0.005] | 1.004 | +0.014 | 不一致 |
| line7 | T2 | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.114 [+0.091, +0.139] | 1.469 | +0.113 | 一致 |
| line7 | T2 | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.242 [+0.208, +0.277] | — | +0.203 | 一致 |
| line7 | POS_STEP | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.000 [+0.000, +0.000] | 1.000 | +0.000 | 一致 |
| line7 | POS_STEP | attach-center | estim-reset-yawlim | estim-yawlim | +0.010 [+0.010, +0.010] | 1.016 | +0.010 | 一致 |
| line7 | POS_STEP | attach-center | estim-yawlim | B1-QP-yawlim | +0.155 [+0.112, +0.201] | 1.327 | +0.163 | 一致 |
| line7 | POS_STEP | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.001, +0.002] | 1.002 | +0.002 | 一致 |
| line7 | POS_STEP | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.099 [+0.094, +0.105] | 1.184 | +0.102 | 一致 |
| line7 | POS_STEP | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.539 [+0.533, +0.545] | — | +0.537 | 一致 |
| line7 | POS_STEP | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.001, +0.002] | 1.003 | +0.002 | 一致 |
| line7 | POS_STEP | drop-center | estim-reset-yawlim | estim-yawlim | +0.010 [+0.009, +0.010] | 1.016 | +0.010 | 一致 |
| line7 | POS_STEP | drop-center | estim-yawlim | B1-QP-yawlim | +0.227 [+0.179, +0.276] | 1.570 | +0.253 | 一致 |
| line7 | POS_STEP | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.015 [+0.011, +0.019] | 1.024 | +0.016 | 一致 |
| line7 | POS_STEP | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.065 [+0.061, +0.069] | 1.121 | +0.063 | 一致 |
| line7 | POS_STEP | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.539 [+0.530, +0.547] | — | +0.538 | 一致 |
| line7 | HOVER | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.002] | 1.002 | +0.002 | 一致 |
| line7 | HOVER | attach-center | estim-reset-yawlim | estim-yawlim | +0.070 [+0.070, +0.071] | 1.077 | +0.070 | 一致 |
| line7 | HOVER | attach-center | estim-yawlim | B1-QP-yawlim | +0.272 [+0.210, +0.338] | 1.422 | +0.294 | 一致 |
| line7 | HOVER | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.014 [+0.013, +0.014] | 1.014 | +0.014 | 一致 |
| line7 | HOVER | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.238 [+0.230, +0.247] | 1.329 | +0.239 | 一致 |
| line7 | HOVER | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.723 [+0.714, +0.732] | — | +0.723 | 一致 |
| line7 | HOVER | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.003] | 1.002 | +0.002 | 一致 |
| line7 | HOVER | drop-center | estim-reset-yawlim | estim-yawlim | +0.071 [+0.070, +0.072] | 1.077 | +0.070 | 一致 |
| line7 | HOVER | drop-center | estim-yawlim | B1-QP-yawlim | +0.291 [+0.226, +0.358] | 1.468 | +0.316 | 一致 |
| line7 | HOVER | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.009 [+0.009, +0.010] | 1.010 | +0.009 | 一致 |
| line7 | HOVER | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.176 [+0.171, +0.180] | 1.227 | +0.172 | 一致 |
| line7 | HOVER | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.772 [+0.768, +0.776] | — | +0.776 | 一致 |
| line8 | T2 | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.000, +0.002] | 1.003 | +0.001 | 一致 |
| line8 | T2 | attach-center | estim-reset-yawlim | estim-yawlim | +0.051 [+0.030, +0.075] | 1.161 | +0.018 | 一致 |
| line8 | T2 | attach-center | estim-yawlim | B1-QP-yawlim | +0.131 [+0.099, +0.164] | 1.701 | +0.125 | 一致 |
| line8 | T2 | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.008 [+0.003, +0.014] | 1.025 | +0.010 | 一致 |
| line8 | T2 | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.125 [+0.096, +0.152] | 1.640 | +0.129 | 一致 |
| line8 | T2 | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.195 [+0.158, +0.235] | — | +0.173 | 一致 |
| line8 | T2 | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.003 [+0.001, +0.008] | 1.009 | +0.001 | 不一致 |
| line8 | T2 | drop-center | estim-reset-yawlim | estim-yawlim | +0.024 [+0.010, +0.041] | 1.067 | +0.020 | 一致 |
| line8 | T2 | drop-center | estim-yawlim | B1-QP-yawlim | +0.179 [+0.140, +0.221] | 2.037 | +0.141 | 一致 |
| line8 | T2 | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.007 [+0.003, +0.013] | 1.022 | +0.008 | 一致 |
| line8 | T2 | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.112 [+0.090, +0.135] | 1.502 | +0.121 | 一致 |
| line8 | T2 | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.223 [+0.189, +0.259] | — | +0.182 | 一致 |
| line8 | POS_STEP | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.000 [+0.000, +0.001] | 1.001 | +0.000 | 一致 |
| line8 | POS_STEP | attach-center | estim-reset-yawlim | estim-yawlim | +0.017 [+0.010, +0.030] | 1.027 | +0.010 | 一致 |
| line8 | POS_STEP | attach-center | estim-yawlim | B1-QP-yawlim | +0.154 [+0.110, +0.200] | 1.331 | +0.167 | 一致 |
| line8 | POS_STEP | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.010 [+0.002, +0.022] | 1.016 | +0.003 | 一致 |
| line8 | POS_STEP | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.104 [+0.095, +0.114] | 1.200 | +0.113 | 一致 |
| line8 | POS_STEP | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.519 [+0.499, +0.535] | — | +0.523 | 一致 |
| line8 | POS_STEP | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.003] | 1.004 | +0.002 | 不一致 |
| line8 | POS_STEP | drop-center | estim-reset-yawlim | estim-yawlim | +0.009 [+0.009, +0.010] | 1.015 | +0.009 | 一致 |
| line8 | POS_STEP | drop-center | estim-yawlim | B1-QP-yawlim | +0.256 [+0.202, +0.308] | 1.709 | +0.272 | 一致 |
| line8 | POS_STEP | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.022 [+0.016, +0.031] | 1.038 | +0.022 | 一致 |
| line8 | POS_STEP | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.065 [+0.060, +0.070] | 1.127 | +0.062 | 一致 |
| line8 | POS_STEP | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.511 [+0.495, +0.524] | — | +0.518 | 一致 |
| line8 | HOVER | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.002] | 1.002 | +0.002 | 一致 |
| line8 | HOVER | attach-center | estim-reset-yawlim | estim-yawlim | +0.069 [+0.068, +0.070] | 1.075 | +0.069 | 一致 |
| line8 | HOVER | attach-center | estim-yawlim | B1-QP-yawlim | +0.260 [+0.199, +0.325] | 1.396 | +0.252 | 一致 |
| line8 | HOVER | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.014 [+0.013, +0.014] | 1.014 | +0.013 | 一致 |
| line8 | HOVER | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.238 [+0.230, +0.247] | 1.330 | +0.236 | 一致 |
| line8 | HOVER | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.723 [+0.714, +0.731] | — | +0.725 | 一致 |
| line8 | HOVER | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.002] | 1.002 | +0.002 | 一致 |
| line8 | HOVER | drop-center | estim-reset-yawlim | estim-yawlim | +0.070 [+0.069, +0.071] | 1.076 | +0.069 | 一致 |
| line8 | HOVER | drop-center | estim-yawlim | B1-QP-yawlim | +0.274 [+0.210, +0.340] | 1.428 | +0.269 | 一致 |
| line8 | HOVER | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.009 [+0.009, +0.010] | 1.010 | +0.009 | 一致 |
| line8 | HOVER | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.174 [+0.170, +0.179] | 1.225 | +0.172 | 一致 |
| line8 | HOVER | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.773 [+0.769, +0.777] | — | +0.776 | 一致 |
| grid10 | T2 | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.001, +0.002] | 1.003 | +0.001 | 一致 |
| grid10 | T2 | attach-center | estim-reset-yawlim | estim-yawlim | +0.044 [+0.029, +0.061] | 1.127 | +0.028 | 一致 |
| grid10 | T2 | attach-center | estim-yawlim | B1-QP-yawlim | +0.227 [+0.195, +0.259] | 2.857 | +0.202 | 一致 |
| grid10 | T2 | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.000, +0.002] | 1.003 | -0.001 | 一致 |
| grid10 | T2 | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.050 [+0.035, +0.067] | 1.152 | +0.047 | 一致 |
| grid10 | T2 | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.221 [+0.182, +0.261] | 3.063 | +0.217 | 一致 |
| grid10 | T2 | drop-center | informed-event-yawlim | estim-reset-yawlim | -0.002 [-0.008, +0.002] | 0.996 | +0.001 | 一致 |
| grid10 | T2 | drop-center | estim-reset-yawlim | estim-yawlim | +0.019 [+0.010, +0.031] | 1.052 | +0.029 | 一致 |
| grid10 | T2 | drop-center | estim-yawlim | B1-QP-yawlim | +0.253 [+0.216, +0.287] | 3.175 | +0.206 | 一致 |
| grid10 | T2 | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.000, +0.004] | 1.005 | +0.002 | 不一致 |
| grid10 | T2 | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.023 [+0.013, +0.034] | 1.063 | +0.032 | 一致 |
| grid10 | T2 | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.265 [+0.229, +0.303] | 3.831 | +0.230 | 一致 |
| grid10 | POS_STEP | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.000 [+0.000, +0.000] | 1.001 | +0.001 | 一致 |
| grid10 | POS_STEP | attach-center | estim-reset-yawlim | estim-yawlim | +0.018 [+0.018, +0.019] | 1.029 | +0.018 | 一致 |
| grid10 | POS_STEP | attach-center | estim-yawlim | B1-QP-yawlim | +0.267 [+0.227, +0.307] | 1.750 | +0.270 | 一致 |
| grid10 | POS_STEP | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.001 [+0.001, +0.001] | 1.001 | +0.001 | 一致 |
| grid10 | POS_STEP | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.023 [+0.021, +0.027] | 1.038 | +0.024 | 一致 |
| grid10 | POS_STEP | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.367 [+0.315, +0.420] | 2.459 | +0.397 | 一致 |
| grid10 | POS_STEP | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.002 [+0.002, +0.002] | 1.003 | +0.002 | 不一致 |
| grid10 | POS_STEP | drop-center | estim-reset-yawlim | estim-yawlim | +0.017 [+0.016, +0.017] | 1.027 | +0.017 | 一致 |
| grid10 | POS_STEP | drop-center | estim-yawlim | B1-QP-yawlim | +0.405 [+0.363, +0.447] | 2.885 | +0.445 | 一致 |
| grid10 | POS_STEP | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.005 [+0.003, +0.008] | 1.008 | +0.006 | 一致 |
| grid10 | POS_STEP | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.018 [+0.017, +0.019] | 1.030 | +0.019 | 一致 |
| grid10 | POS_STEP | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.418 [+0.372, +0.466] | 3.146 | +0.431 | 一致 |
| grid10 | HOVER | attach-center | informed-event-yawlim | estim-reset-yawlim | +0.005 [+0.005, +0.005] | 1.005 | +0.005 | 一致 |
| grid10 | HOVER | attach-center | estim-reset-yawlim | estim-yawlim | +0.096 [+0.095, +0.097] | 1.109 | +0.096 | 一致 |
| grid10 | HOVER | attach-center | estim-yawlim | B1-QP-yawlim | +0.409 [+0.358, +0.462] | 1.862 | +0.412 | 一致 |
| grid10 | HOVER | attach-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.007 [+0.006, +0.007] | 1.007 | +0.006 | 一致 |
| grid10 | HOVER | attach-offcenter | estim-reset-yawlim | estim-yawlim | +0.106 [+0.100, +0.112] | 1.121 | +0.104 | 一致 |
| grid10 | HOVER | attach-offcenter | estim-yawlim | B1-QP-yawlim | +0.556 [+0.485, +0.623] | 2.757 | +0.543 | 一致 |
| grid10 | HOVER | drop-center | informed-event-yawlim | estim-reset-yawlim | +0.003 [+0.003, +0.003] | 1.003 | +0.003 | 一致 |
| grid10 | HOVER | drop-center | estim-reset-yawlim | estim-yawlim | +0.088 [+0.088, +0.089] | 1.100 | +0.088 | 一致 |
| grid10 | HOVER | drop-center | estim-yawlim | B1-QP-yawlim | +0.463 [+0.407, +0.519] | 2.094 | +0.467 | 一致 |
| grid10 | HOVER | drop-offcenter | informed-event-yawlim | estim-reset-yawlim | +0.005 [+0.004, +0.006] | 1.005 | +0.005 | 一致 |
| grid10 | HOVER | drop-offcenter | estim-reset-yawlim | estim-yawlim | +0.091 [+0.088, +0.094] | 1.103 | +0.090 | 一致 |
| grid10 | HOVER | drop-offcenter | estim-yawlim | B1-QP-yawlim | +0.578 [+0.508, +0.647] | 2.908 | +0.562 | 一致 |

### EV 其他指標（均值；峰值 = 事件後 1 s 位置誤差峰值 m；恢復 = 恢復時間中位數 s，未恢復以剩餘時間計）

| 構型 | 任務 | 條件 | 方法 | 峰值 | 恢復(s) | 恢復比例 | 探索 seed 峰值 |
|---|---|---|---|---|---|---|---|
| line6 | T2 | attach-center | B1-QP-yawlim | 1.380 | 1.88 | 0.40 | 1.404 |
| line6 | T2 | attach-center | estim-yawlim | 1.248 | 1.27 | 0.53 | 1.330 |
| line6 | T2 | attach-center | estim-reset-yawlim | 1.222 | 0.94 | 0.61 | 1.324 |
| line6 | T2 | attach-center | informed-event-yawlim | 1.213 | 0.94 | 0.61 | 1.324 |
| line6 | T2 | attach-offcenter | B1-QP-yawlim | 2.469 | 2.73 | 0.00 | 2.448 |
| line6 | T2 | attach-offcenter | estim-yawlim | 1.553 | 2.05 | 0.40 | 1.587 |
| line6 | T2 | attach-offcenter | estim-reset-yawlim | 1.246 | 1.17 | 0.54 | 1.340 |
| line6 | T2 | attach-offcenter | informed-event-yawlim | 1.240 | 1.05 | 0.57 | 1.333 |
| line6 | T2 | drop-center | B1-QP-yawlim | 2.080 | 1.01 | 0.55 | 2.038 |
| line6 | T2 | drop-center | estim-yawlim | 1.340 | 0.81 | 0.68 | 1.367 |
| line6 | T2 | drop-center | estim-reset-yawlim | 1.309 | 0.65 | 0.74 | 1.353 |
| line6 | T2 | drop-center | informed-event-yawlim | 1.309 | 0.66 | 0.74 | 1.364 |
| line6 | T2 | drop-offcenter | B1-QP-yawlim | 3.001 | 2.73 | 0.00 | 3.001 |
| line6 | T2 | drop-offcenter | estim-yawlim | 1.602 | 1.49 | 0.56 | 1.659 |
| line6 | T2 | drop-offcenter | estim-reset-yawlim | 1.397 | 0.15 | 0.76 | 1.452 |
| line6 | T2 | drop-offcenter | informed-event-yawlim | 1.366 | 0.31 | 0.75 | 1.428 |
| line6 | POS_STEP | attach-center | B1-QP-yawlim | 0.467 | 2.45 | 0.22 | 0.476 |
| line6 | POS_STEP | attach-center | estim-yawlim | 0.267 | 2.44 | 0.27 | 0.265 |
| line6 | POS_STEP | attach-center | estim-reset-yawlim | 0.083 | 0.86 | 0.99 | 0.069 |
| line6 | POS_STEP | attach-center | informed-event-yawlim | 0.076 | 0.49 | 1.00 | 0.062 |
| line6 | POS_STEP | attach-offcenter | B1-QP-yawlim | 2.334 | 2.71 | 0.00 | 2.333 |
| line6 | POS_STEP | attach-offcenter | estim-yawlim | 0.966 | 2.44 | 0.26 | 0.974 |
| line6 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.174 | 1.45 | 0.81 | 0.169 |
| line6 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.125 | 1.28 | 0.88 | 0.114 |
| line6 | POS_STEP | drop-center | B1-QP-yawlim | 1.330 | 0.00 | 0.70 | 1.312 |
| line6 | POS_STEP | drop-center | estim-yawlim | 0.253 | 2.38 | 0.25 | 0.249 |
| line6 | POS_STEP | drop-center | estim-reset-yawlim | 0.086 | 0.82 | 0.99 | 0.074 |
| line6 | POS_STEP | drop-center | informed-event-yawlim | 0.082 | 0.68 | 0.99 | 0.071 |
| line6 | POS_STEP | drop-offcenter | B1-QP-yawlim | 3.000 | 2.71 | 0.00 | 3.000 |
| line6 | POS_STEP | drop-offcenter | estim-yawlim | 0.821 | 1.90 | 0.39 | 0.822 |
| line6 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.218 | 1.14 | 0.88 | 0.204 |
| line6 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.184 | 1.26 | 0.85 | 0.179 |
| line6 | HOVER | attach-center | B1-QP-yawlim | 0.455 | 2.68 | 0.00 | 0.470 |
| line6 | HOVER | attach-center | estim-yawlim | 0.256 | 2.68 | 0.01 | 0.259 |
| line6 | HOVER | attach-center | estim-reset-yawlim | 0.049 | 1.01 | 0.98 | 0.051 |
| line6 | HOVER | attach-center | informed-event-yawlim | 0.041 | 0.85 | 1.00 | 0.042 |
| line6 | HOVER | attach-offcenter | B1-QP-yawlim | 2.322 | 2.68 | 0.00 | 2.323 |
| line6 | HOVER | attach-offcenter | estim-yawlim | 0.943 | 2.68 | 0.00 | 0.946 |
| line6 | HOVER | attach-offcenter | estim-reset-yawlim | 0.159 | 1.48 | 0.80 | 0.159 |
| line6 | HOVER | attach-offcenter | informed-event-yawlim | 0.102 | 1.31 | 0.87 | 0.102 |
| line6 | HOVER | drop-center | B1-QP-yawlim | 1.127 | 0.00 | 0.84 | 1.155 |
| line6 | HOVER | drop-center | estim-yawlim | 0.242 | 2.68 | 0.00 | 0.244 |
| line6 | HOVER | drop-center | estim-reset-yawlim | 0.051 | 0.99 | 0.98 | 0.053 |
| line6 | HOVER | drop-center | informed-event-yawlim | 0.047 | 0.94 | 0.98 | 0.049 |
| line6 | HOVER | drop-offcenter | B1-QP-yawlim | 3.000 | 2.68 | 0.00 | 3.001 |
| line6 | HOVER | drop-offcenter | estim-yawlim | 0.810 | 2.38 | 0.15 | 0.802 |
| line6 | HOVER | drop-offcenter | estim-reset-yawlim | 0.156 | 1.35 | 0.86 | 0.155 |
| line6 | HOVER | drop-offcenter | informed-event-yawlim | 0.151 | 1.37 | 0.84 | 0.151 |
| line7 | T2 | attach-center | B1-QP-yawlim | 1.385 | 2.05 | 0.36 | 1.397 |
| line7 | T2 | attach-center | estim-yawlim | 1.288 | 1.29 | 0.52 | 1.344 |
| line7 | T2 | attach-center | estim-reset-yawlim | 1.235 | 1.02 | 0.57 | 1.333 |
| line7 | T2 | attach-center | informed-event-yawlim | 1.234 | 1.00 | 0.58 | 1.332 |
| line7 | T2 | attach-offcenter | B1-QP-yawlim | 2.473 | 2.73 | 0.00 | 2.462 |
| line7 | T2 | attach-offcenter | estim-yawlim | 1.586 | 2.02 | 0.40 | 1.628 |
| line7 | T2 | attach-offcenter | estim-reset-yawlim | 1.257 | 1.67 | 0.51 | 1.353 |
| line7 | T2 | attach-offcenter | informed-event-yawlim | 1.250 | 1.15 | 0.55 | 1.344 |
| line7 | T2 | drop-center | B1-QP-yawlim | 2.039 | 0.90 | 0.58 | 2.066 |
| line7 | T2 | drop-center | estim-yawlim | 1.346 | 0.79 | 0.71 | 1.381 |
| line7 | T2 | drop-center | estim-reset-yawlim | 1.313 | 0.61 | 0.73 | 1.362 |
| line7 | T2 | drop-center | informed-event-yawlim | 1.300 | 0.60 | 0.74 | 1.371 |
| line7 | T2 | drop-offcenter | B1-QP-yawlim | 3.000 | 2.73 | 0.00 | 3.002 |
| line7 | T2 | drop-offcenter | estim-yawlim | 1.618 | 1.39 | 0.58 | 1.731 |
| line7 | T2 | drop-offcenter | estim-reset-yawlim | 1.405 | 0.00 | 0.77 | 1.530 |
| line7 | T2 | drop-offcenter | informed-event-yawlim | 1.407 | 0.28 | 0.75 | 1.481 |
| line7 | POS_STEP | attach-center | B1-QP-yawlim | 0.440 | 2.45 | 0.22 | 0.451 |
| line7 | POS_STEP | attach-center | estim-yawlim | 0.262 | 2.44 | 0.28 | 0.260 |
| line7 | POS_STEP | attach-center | estim-reset-yawlim | 0.084 | 0.88 | 0.99 | 0.072 |
| line7 | POS_STEP | attach-center | informed-event-yawlim | 0.078 | 0.65 | 1.00 | 0.065 |
| line7 | POS_STEP | attach-offcenter | B1-QP-yawlim | 2.313 | 2.71 | 0.00 | 2.306 |
| line7 | POS_STEP | attach-offcenter | estim-yawlim | 0.965 | 2.42 | 0.32 | 0.968 |
| line7 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.173 | 1.45 | 0.81 | 0.170 |
| line7 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.124 | 1.28 | 0.88 | 0.116 |
| line7 | POS_STEP | drop-center | B1-QP-yawlim | 1.205 | 0.00 | 0.75 | 1.331 |
| line7 | POS_STEP | drop-center | estim-yawlim | 0.248 | 1.84 | 0.36 | 0.245 |
| line7 | POS_STEP | drop-center | estim-reset-yawlim | 0.097 | 0.00 | 0.99 | 0.081 |
| line7 | POS_STEP | drop-center | informed-event-yawlim | 0.091 | 0.00 | 0.99 | 0.076 |
| line7 | POS_STEP | drop-offcenter | B1-QP-yawlim | 3.001 | 2.71 | 0.00 | 3.000 |
| line7 | POS_STEP | drop-offcenter | estim-yawlim | 0.841 | 1.70 | 0.49 | 0.830 |
| line7 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.261 | 0.00 | 0.87 | 0.238 |
| line7 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.210 | 1.04 | 0.85 | 0.194 |
| line7 | HOVER | attach-center | B1-QP-yawlim | 0.432 | 2.68 | 0.00 | 0.447 |
| line7 | HOVER | attach-center | estim-yawlim | 0.250 | 2.68 | 0.01 | 0.253 |
| line7 | HOVER | attach-center | estim-reset-yawlim | 0.050 | 1.01 | 0.98 | 0.053 |
| line7 | HOVER | attach-center | informed-event-yawlim | 0.043 | 0.89 | 1.00 | 0.045 |
| line7 | HOVER | attach-offcenter | B1-QP-yawlim | 2.299 | 2.68 | 0.00 | 2.297 |
| line7 | HOVER | attach-offcenter | estim-yawlim | 0.938 | 2.68 | 0.00 | 0.941 |
| line7 | HOVER | attach-offcenter | estim-reset-yawlim | 0.159 | 1.47 | 0.80 | 0.159 |
| line7 | HOVER | attach-offcenter | informed-event-yawlim | 0.103 | 1.31 | 0.87 | 0.102 |
| line7 | HOVER | drop-center | B1-QP-yawlim | 1.060 | 0.00 | 0.76 | 1.123 |
| line7 | HOVER | drop-center | estim-yawlim | 0.238 | 2.68 | 0.00 | 0.240 |
| line7 | HOVER | drop-center | estim-reset-yawlim | 0.052 | 0.99 | 0.98 | 0.054 |
| line7 | HOVER | drop-center | informed-event-yawlim | 0.048 | 0.96 | 0.98 | 0.051 |
| line7 | HOVER | drop-offcenter | B1-QP-yawlim | 3.000 | 2.68 | 0.00 | 3.002 |
| line7 | HOVER | drop-offcenter | estim-yawlim | 0.805 | 2.38 | 0.15 | 0.793 |
| line7 | HOVER | drop-offcenter | estim-reset-yawlim | 0.156 | 1.35 | 0.86 | 0.155 |
| line7 | HOVER | drop-offcenter | informed-event-yawlim | 0.151 | 1.37 | 0.84 | 0.151 |
| line8 | T2 | attach-center | B1-QP-yawlim | 1.460 | 2.08 | 0.33 | 1.411 |
| line8 | T2 | attach-center | estim-yawlim | 1.362 | 1.77 | 0.49 | 1.368 |
| line8 | T2 | attach-center | estim-reset-yawlim | 1.268 | 1.17 | 0.59 | 1.352 |
| line8 | T2 | attach-center | informed-event-yawlim | 1.259 | 1.02 | 0.60 | 1.350 |
| line8 | T2 | attach-offcenter | B1-QP-yawlim | 2.477 | 2.73 | 0.00 | 2.470 |
| line8 | T2 | attach-offcenter | estim-yawlim | 1.674 | 2.16 | 0.35 | 1.704 |
| line8 | T2 | attach-offcenter | estim-reset-yawlim | 1.329 | 1.81 | 0.50 | 1.388 |
| line8 | T2 | attach-offcenter | informed-event-yawlim | 1.305 | 1.77 | 0.51 | 1.366 |
| line8 | T2 | drop-center | B1-QP-yawlim | 2.194 | 1.61 | 0.49 | 2.076 |
| line8 | T2 | drop-center | estim-yawlim | 1.351 | 0.70 | 0.71 | 1.427 |
| line8 | T2 | drop-center | estim-reset-yawlim | 1.341 | 0.58 | 0.75 | 1.408 |
| line8 | T2 | drop-center | informed-event-yawlim | 1.332 | 0.59 | 0.77 | 1.406 |
| line8 | T2 | drop-offcenter | B1-QP-yawlim | 3.001 | 2.73 | 0.00 | 3.002 |
| line8 | T2 | drop-offcenter | estim-yawlim | 1.670 | 1.60 | 0.58 | 1.802 |
| line8 | T2 | drop-offcenter | estim-reset-yawlim | 1.470 | 0.17 | 0.75 | 1.570 |
| line8 | T2 | drop-offcenter | informed-event-yawlim | 1.452 | 0.38 | 0.73 | 1.538 |
| line8 | POS_STEP | attach-center | B1-QP-yawlim | 0.478 | 2.45 | 0.22 | 0.450 |
| line8 | POS_STEP | attach-center | estim-yawlim | 0.288 | 2.01 | 0.40 | 0.266 |
| line8 | POS_STEP | attach-center | estim-reset-yawlim | 0.103 | 0.68 | 0.99 | 0.082 |
| line8 | POS_STEP | attach-center | informed-event-yawlim | 0.096 | 0.00 | 1.00 | 0.075 |
| line8 | POS_STEP | attach-offcenter | B1-QP-yawlim | 2.305 | 2.71 | 0.00 | 2.300 |
| line8 | POS_STEP | attach-offcenter | estim-yawlim | 1.006 | 2.32 | 0.32 | 0.992 |
| line8 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.208 | 1.35 | 0.80 | 0.187 |
| line8 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.153 | 1.07 | 0.88 | 0.131 |
| line8 | POS_STEP | drop-center | B1-QP-yawlim | 1.349 | 0.00 | 0.73 | 1.410 |
| line8 | POS_STEP | drop-center | estim-yawlim | 0.259 | 0.78 | 0.50 | 0.248 |
| line8 | POS_STEP | drop-center | estim-reset-yawlim | 0.136 | 0.00 | 1.00 | 0.111 |
| line8 | POS_STEP | drop-center | informed-event-yawlim | 0.130 | 0.00 | 1.00 | 0.105 |
| line8 | POS_STEP | drop-offcenter | B1-QP-yawlim | 3.001 | 2.71 | 0.00 | 3.000 |
| line8 | POS_STEP | drop-offcenter | estim-yawlim | 0.879 | 0.00 | 0.61 | 0.884 |
| line8 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.358 | 0.00 | 0.93 | 0.323 |
| line8 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.276 | 0.00 | 0.87 | 0.241 |
| line8 | HOVER | attach-center | B1-QP-yawlim | 0.414 | 2.68 | 0.00 | 0.414 |
| line8 | HOVER | attach-center | estim-yawlim | 0.247 | 2.68 | 0.01 | 0.248 |
| line8 | HOVER | attach-center | estim-reset-yawlim | 0.052 | 0.99 | 0.98 | 0.054 |
| line8 | HOVER | attach-center | informed-event-yawlim | 0.045 | 0.76 | 1.00 | 0.047 |
| line8 | HOVER | attach-offcenter | B1-QP-yawlim | 2.282 | 2.68 | 0.00 | 2.281 |
| line8 | HOVER | attach-offcenter | estim-yawlim | 0.936 | 2.68 | 0.00 | 0.934 |
| line8 | HOVER | attach-offcenter | estim-reset-yawlim | 0.160 | 1.48 | 0.76 | 0.159 |
| line8 | HOVER | attach-offcenter | informed-event-yawlim | 0.104 | 1.30 | 0.87 | 0.103 |
| line8 | HOVER | drop-center | B1-QP-yawlim | 1.005 | 0.00 | 0.78 | 0.977 |
| line8 | HOVER | drop-center | estim-yawlim | 0.235 | 2.68 | 0.00 | 0.236 |
| line8 | HOVER | drop-center | estim-reset-yawlim | 0.053 | 0.95 | 0.98 | 0.055 |
| line8 | HOVER | drop-center | informed-event-yawlim | 0.050 | 0.86 | 0.98 | 0.052 |
| line8 | HOVER | drop-offcenter | B1-QP-yawlim | 3.000 | 2.68 | 0.00 | 3.002 |
| line8 | HOVER | drop-offcenter | estim-yawlim | 0.801 | 2.35 | 0.15 | 0.793 |
| line8 | HOVER | drop-offcenter | estim-reset-yawlim | 0.156 | 1.35 | 0.87 | 0.155 |
| line8 | HOVER | drop-offcenter | informed-event-yawlim | 0.151 | 1.37 | 0.84 | 0.151 |
| grid10 | T2 | attach-center | B1-QP-yawlim | 1.487 | 2.40 | 0.26 | 1.480 |
| grid10 | T2 | attach-center | estim-yawlim | 1.257 | 1.43 | 0.54 | 1.342 |
| grid10 | T2 | attach-center | estim-reset-yawlim | 1.209 | 1.00 | 0.59 | 1.317 |
| grid10 | T2 | attach-center | informed-event-yawlim | 1.207 | 0.99 | 0.59 | 1.317 |
| grid10 | T2 | attach-offcenter | B1-QP-yawlim | 1.707 | 2.50 | 0.19 | 1.705 |
| grid10 | T2 | attach-offcenter | estim-yawlim | 1.295 | 1.81 | 0.50 | 1.380 |
| grid10 | T2 | attach-offcenter | estim-reset-yawlim | 1.213 | 1.11 | 0.56 | 1.323 |
| grid10 | T2 | attach-offcenter | informed-event-yawlim | 1.210 | 1.10 | 0.56 | 1.321 |
| grid10 | T2 | drop-center | B1-QP-yawlim | 2.394 | 1.78 | 0.46 | 2.266 |
| grid10 | T2 | drop-center | estim-yawlim | 1.309 | 0.55 | 0.72 | 1.376 |
| grid10 | T2 | drop-center | estim-reset-yawlim | 1.291 | 0.56 | 0.75 | 1.338 |
| grid10 | T2 | drop-center | informed-event-yawlim | 1.305 | 0.64 | 0.75 | 1.336 |
| grid10 | T2 | drop-offcenter | B1-QP-yawlim | 2.561 | 2.19 | 0.30 | 2.511 |
| grid10 | T2 | drop-offcenter | estim-yawlim | 1.357 | 0.56 | 0.67 | 1.416 |
| grid10 | T2 | drop-offcenter | estim-reset-yawlim | 1.325 | 0.40 | 0.74 | 1.369 |
| grid10 | T2 | drop-offcenter | informed-event-yawlim | 1.310 | 0.38 | 0.71 | 1.367 |
| grid10 | POS_STEP | attach-center | B1-QP-yawlim | 0.814 | 2.47 | 0.19 | 0.831 |
| grid10 | POS_STEP | attach-center | estim-yawlim | 0.397 | 2.44 | 0.27 | 0.400 |
| grid10 | POS_STEP | attach-center | estim-reset-yawlim | 0.109 | 1.23 | 0.89 | 0.095 |
| grid10 | POS_STEP | attach-center | informed-event-yawlim | 0.093 | 1.09 | 0.98 | 0.078 |
| grid10 | POS_STEP | attach-offcenter | B1-QP-yawlim | 1.020 | 2.67 | 0.07 | 1.076 |
| grid10 | POS_STEP | attach-offcenter | estim-yawlim | 0.434 | 2.44 | 0.27 | 0.448 |
| grid10 | POS_STEP | attach-offcenter | estim-reset-yawlim | 0.115 | 1.12 | 0.92 | 0.106 |
| grid10 | POS_STEP | attach-offcenter | informed-event-yawlim | 0.094 | 0.96 | 0.96 | 0.082 |
| grid10 | POS_STEP | drop-center | B1-QP-yawlim | 1.989 | 0.00 | 0.57 | 2.090 |
| grid10 | POS_STEP | drop-center | estim-yawlim | 0.359 | 2.44 | 0.24 | 0.358 |
| grid10 | POS_STEP | drop-center | estim-reset-yawlim | 0.113 | 1.13 | 0.93 | 0.100 |
| grid10 | POS_STEP | drop-center | informed-event-yawlim | 0.108 | 1.16 | 0.94 | 0.095 |
| grid10 | POS_STEP | drop-offcenter | B1-QP-yawlim | 2.127 | 1.70 | 0.45 | 2.194 |
| grid10 | POS_STEP | drop-offcenter | estim-yawlim | 0.377 | 2.14 | 0.31 | 0.393 |
| grid10 | POS_STEP | drop-offcenter | estim-reset-yawlim | 0.131 | 1.03 | 0.97 | 0.137 |
| grid10 | POS_STEP | drop-offcenter | informed-event-yawlim | 0.114 | 1.03 | 0.96 | 0.113 |
| grid10 | HOVER | attach-center | B1-QP-yawlim | 0.804 | 2.68 | 0.00 | 0.814 |
| grid10 | HOVER | attach-center | estim-yawlim | 0.391 | 2.68 | 0.00 | 0.393 |
| grid10 | HOVER | attach-center | estim-reset-yawlim | 0.079 | 1.26 | 0.89 | 0.079 |
| grid10 | HOVER | attach-center | informed-event-yawlim | 0.060 | 1.09 | 0.98 | 0.060 |
| grid10 | HOVER | attach-offcenter | B1-QP-yawlim | 1.038 | 2.68 | 0.00 | 1.013 |
| grid10 | HOVER | attach-offcenter | estim-yawlim | 0.433 | 2.68 | 0.00 | 0.427 |
| grid10 | HOVER | attach-offcenter | estim-reset-yawlim | 0.088 | 1.32 | 0.91 | 0.087 |
| grid10 | HOVER | attach-offcenter | informed-event-yawlim | 0.063 | 1.17 | 0.95 | 0.062 |
| grid10 | HOVER | drop-center | B1-QP-yawlim | 1.592 | 0.00 | 0.76 | 1.592 |
| grid10 | HOVER | drop-center | estim-yawlim | 0.353 | 2.68 | 0.09 | 0.354 |
| grid10 | HOVER | drop-center | estim-reset-yawlim | 0.082 | 1.19 | 0.90 | 0.083 |
| grid10 | HOVER | drop-center | informed-event-yawlim | 0.078 | 1.17 | 0.93 | 0.078 |
| grid10 | HOVER | drop-offcenter | B1-QP-yawlim | 2.040 | 1.88 | 0.42 | 1.970 |
| grid10 | HOVER | drop-offcenter | estim-yawlim | 0.370 | 2.57 | 0.05 | 0.365 |
| grid10 | HOVER | drop-offcenter | estim-reset-yawlim | 0.085 | 1.12 | 0.95 | 0.085 |
| grid10 | HOVER | drop-offcenter | informed-event-yawlim | 0.083 | 1.15 | 0.93 | 0.083 |

註：T2 的峰值與恢復時間混入換目標暫態（prereg 5.c 已知限制）。
