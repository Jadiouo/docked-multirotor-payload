# G2 狀態（中止於 2026-10-25，使用者要求收尾）

## 已完成並驗證
- quadsim/controllers.py：B3 INDI、B3+adapt（G 對角縮放 RLS）、B1-QP+adapt、B4 簡化版（B1-QP+adapt + 力矩偏置 b̂ 代 CoM 偏移）、FzAdapter（總推力增益，加速度計）。B1QP 內部重構成 `_solve`，行為不變。
- scripts/g2_mismatch.py、g1_gate.run_cell 加選用參數 perturb/ckw（預設不變）。tests/test_controllers.py 新增 4 個測試。
- pytest tests/test_controllers.py tests/test_env.py：42 passed。

## g2_raw.json（n=100）已跑 749/864 格
nominal、payload10/15/30、thrust10/20、dock 全部完成；all 完成 77/96；all15（補充）尚未跑。

## 部分結果（報告 results/p2/g2.md 是 --report-only 產生，all 不完整）
- 推力 ±20%、對接偏差：最強古典法僅掉 0.3 到 1.7 pp。
- payload 最傷：10% 最強（B3+adapt）掉 5.6 pp，15% 最強（INDI）掉 15 pp；B1-QP 掉 51/90 pp。
- payload 30% 在 TWR 1.6 下連懸停都不可行，全部方法 0%。

## 續跑
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 .venv/bin/python -m scripts.g2_mismatch --n-eps 100`（自動續跑缺的格，約 40 分鐘）。判定文字解讀尚未寫（放 results/p2/g2_notes.md）。
