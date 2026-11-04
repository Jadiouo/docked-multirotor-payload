# P3 RL 管線狀態（收尾）

已完成並驗證
- tests/test_policy.py：11 passed（CPU）。
- env 在 cuda（B=4096，load≈73）：N=1 359k、N=4 2059k、N=8 3260k module-steps/s，env.py 無報錯；受 launch 開銷限制（約 8-11 ms/step 與 B 無關）。
- 單機 T1（1.5e8 步，11 分）：成功率 100%，分數對 B1：ATT 1.057、POS 1.078、T2 1.170。
- 單機 T2（同量）：T2 成功率 99%、分數 0.398（B1 的 1.027）；ATT 0.597、POS 0.951。訓練 reward 停在約 0，T2 專訓反而不如 T1 策略（每步 reward 0.018 對 0.143），判斷為最佳化卡住。
- 多構型煙霧（k=1,H=4，5 分）：loss 有限，step_reward -0.95 到 -0.48；pad/mask/buffer 跑通。

進行到一半
- results/p3/train4_k1_h4（TRAIN 四構型，約 20 分，到第 265 輪成功率 50%，step_reward 0.60）被取消，只有 latest.pt，沒有 final.pt、未 eval。
- #108（single_T1T2 混合）、#109（single_T2_ent，ent 0.005）已取消，未跑。

下次接手
- 先跑 #108/#109 找出 T2 專訓卡住的原因；再跑 train4 完整版，用 rl_eval 評估。

已知問題
- ppo.py 以單模組為樣本，每筆樣本帶整架資料，造成 N_max 倍重複計算與記憶體。
- gpujob 隊列常等數十分鐘到一小時。


## 2026-10-30 更新（T2 診斷、team 共用、train4 重跑排隊中）

### T2 專訓卡住的診斷（診斷腳本 scripts/p3_diag.py；都在 T2 環境、256 個 episode、deterministic）
| 策略 | 每步 reward | 平均位置誤差 | 平均角速度 | 動作飽和比例 | 存活 |
|---|---|---|---|---|---|
| single_T1（只練 T1） | 0.134 | 0.61 m | 1.30 | 6% | 99.96% |
| single_T2（只練 T2） | 0.012 | 0.77 m | 1.93 | 4% | 99.9% |
| diag_ent005（T2，ent 0.005） | 0.021 | 0.77 m | 0.83 | 16% | 100% |
| diag_mix_T1T2（T1+T2 各半） | 0.091 | 0.68 m | 1.25 | 5% | 99.96% |
- 不是存活問題（終止原因幾乎全是 truncated），也不是飽和或 log_std 過早塌縮（T2 專訓 log_std -2.2，比 T1 的 -3.0 還大）。
- T2 reward 的常數項 alive=1 與 -|e_p|（平均約 0.6–0.8）、-0.2|e_yaw| 幾乎抵消，reward 在 0 附近，梯度訊號弱；T2 專訓停在「穩定 hover + 慢追蹤」平台（step_reward 從第 140 輪起 ±0.03 不動）。
- ent_coef=0.005（diag_T2_ent005）：reward 仍 0.021，沒有脫離平台，只多了飽和 → 不是探索不足。
- T1+T2 混合（diag_mix_T1T2，單機 k=0 H=1，1.5e8 步）：eval 比值 vs B1：ATT_STEP 1.042、POS_STEP 1.044、T2 1.091（vs max(B1,QP) 1.049），成功率 100/100/98%。T1 提供的大擾動回復訊號給了策略追蹤所需的敏捷度。
- reward 與分數對齊：score = success×(1−e_rms/e_ref)（RMS，ATT_STEP 用姿態誤差），reward 用 L1 位置誤差、沒有姿態誤差項。ATT_STEP 只靠 rate 與位置項間接約束，但單機實測已 ≥1.04，故**沒有改 reward 預設值**（env.py 沒動）。
- 結論：T2 專訓是最佳化平台，不是 bug；解法是 T1+T2 混合（curriculum 的簡化版）。

### team 共用（不改 ppo.py）
- 新增 ModularActorCritic.act_team / value_team：rollout 時同一環境 n_max 列資料相同，GNN 只算一次；rl_train 的 rollout（act、兩次 V）改用它。tests/test_policy.py::test_team_entry_points_match_per_row 驗證與逐列 act/value 數值一致（CPU：39 passed）。
- CPU 微基準（n_max=4、H=4、k=1、2048 env）：act 119→17 ms（7 倍）、value 105→8 ms（13 倍）。
- 做不到的部分：ppo_update 從 flat 樣本隨機抽「單列」，每列仍自帶整隊並重算 GNN（update 的 n_max 倍重複），同一 minibatch 內同隊兄弟列平均不到 0.25 個，去重沒用。根治要把樣本單位改成「整隊」（joint 動作、joint ratio），這會改變演算法（ratio 變成 N 個模組乘積、clip 更嚴），需要使用者決定，我沒做。

### train4 重跑
- #55：results/p3/train4_mix_k1_h4（單機/line2/line3/square4，各 T1 與 T2 一組，每組 512 環境，k=1、H=4，3e8 步，上限 170 分鐘）。截至本次結束仍在 gpujob 隊列中（前面有 ctde-mappo-vmas 的約 24 個短工作，估計數小時），**尚未跑、尚未 eval**。
- 跑完後評估：
  CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.rl_eval results/p3/train4_mix_k1_h4/final.pt --configs single line2 line3 square4
  （完成前 latest.pt 可先用同指令評估。）
- 舊的 train4_k1_h4（只練 T1、75M 步，成功率 51%）：square4 成功率只有 25%，k=1/H=4 的多構型學得慢，單純需要更多步數，是否需要再調（例如 lr、n_epochs）要看 #55 的曲線。
- 其他：results/p2/g2_raw.json 在 git status 顯示有修改，不是我改的（G2 背景腳本）。

### line8 RL go/no-go 先導（PLAN v5.1，協調者新增）
- 已排隊 #71（在 #55 之後）：results/p3/line8_mix_k2_h4 = line8:T1 + line8:T2 各 512 環境、k=2、H=4、1.5e8 步、time-limit 300 分鐘、訓練 --seed 7（環境 seed 7000/7001，與 rl_eval 用的 g1 SEED 0-3 分開）。VRAM 估計約 4GB。
- 評估指令：CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.rl_eval results/p3/line8_mix_k2_h4/final.pt --configs line8 --tasks ATT_STEP POS_STEP T2
- 比較基準（results/p2/g1_v2.md，line8，標稱）：分母 max(B1,B1-QP)：T2 0.274（B1-QP，成功率 80%）、POS_STEP 0.640、ATT_STEP 0.678；最強局部古典法（B2-QP+adapt k=2）：T2 0.137（46%）、POS_STEP 0.587、ATT_STEP 見 g1_v2.md 第 100 行（adapt k=2 落後 5.4pp）。
- 判定問題：RL 能否學會 line8（成功率、分數）；T2 分數落在 0.137（局部古典）與 0.274（分母）之間的哪裡。尚未有結果。

### 未知 payload 先導（line6，payload DR）
- 程式：quadsim/rl_env.py 的 RandomPayloadEnv（每環境每次 reset 抽質量比例 U(0, pmax)、掛點模組在所有模組均勻抽、偏移 0）；rl_train `--payload-max`（預設 0 = 關閉，行為不變）。actor 看不到 payload；critic 特權欄位仍是原本的 d（含 payload 的真實 CoM 相對位置），維度沒變，舊 checkpoint 可直接載入。**沒有加質量比例／掛點的顯式特權欄位**，那要改 policy.py 的 PRIV_DIM（4→8），未動，待決定。
- 已排隊 #73：results/p3/line6_pl20_k2_h4 = line6:T1 + line6:T2 各 512 環境、k=2、H=4、1.5e8 步、time-limit 300 分鐘、--seed 7（環境 seed 7000/7001）、--payload-max 0.20。VRAM 估計約 4GB。（原派工單寫 pl15／0.15，指揮官補充後改為 0.20 DR，所以名稱是 pl20；未送過舊版。）
- 評估指令（CPU）：
  E1（g2 固定點：15%、argmax-x 掛點，同 seed，可對照既有古典數字）：
  CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.rl_eval results/p3/line6_pl20_k2_h4/final.pt --configs line6 --tasks T2 POS_STEP --payload payload15 --out results/p3/line6_pl20_k2_h4/eval_E1.json
  E2（留出集：質量 {0.10,0.15,0.20} x 隨機掛點，seed 固定為 g1 SEED[task]，與訓練分開）：
  CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.rl_eval results/p3/line6_pl20_k2_h4/final.pt --configs line6 --tasks T2 POS_STEP --holdout --out results/p3/line6_pl20_k2_h4/eval_E2.json
- E1 比較基準（line6、payload15、T2，每格 100 episode；出處 results/p2/g2_informed.md 第 52 行表列與 g2_raw.json、g2_informed_raw.json）：B1 0.000（0%）、B1-QP 0.002（2%）、B3 INDI 0.176（56%）、B3+adapt 0.178（61%，最強古典法）、B1-QP-informed 0.210（69%，上限）。
- E2 基準：無既有數字；rl_eval 首次執行時用同 seed 的 RandomPayloadEnv 重算 B1、B1-QP、B3 系列，快取在 results/p3/rl_eval_cache.json（key 含條件）。冒煙測試時算出的 T2 數字（100 episode）：frac 0.10 → B1 0.106、B1-QP 0.216、B3 INDI 0.296；0.15 → B1-QP 0.094、B3 INDI 0.247；0.20 → B1-QP 0.034、B3+adapt 0.145。B1-QP-informed 在 E2 不支援（需逐 episode host，只在 E1）。
- 可行性：line6、line8 在 20% payload 掛任一模組，懸停 LP 都可行（推重比 1.6）。
- 判定問題：RL（未知 payload）在 E1 T2 是否超過 B3+adapt 0.178、離 informed 0.210 多遠；E2 是否維持。尚未有結果。


### 2026-10-30 RQ-D 準備：RL DR 擴充與先導重送（#73 取消）
- 程式：quadsim/env.py 新增 `Perturb.payload_event`（中途 attach／drop，連續性與限制寫在 ModularEnv 的「payload 事件」註解）；RandomPayloadEnv 新增 `offset_max`、`event_prob`；rl_train 新增 `--payload-offset-max s`（x、y 各 U(−s·0.0333, s·0.0333) m，z U(−s·0.0215, 0) m）、`--payload-event-prob p`（每 episode 以機率 p 改成 attach／drop 各半的事件，事件時刻 U(1.0, 3.5) s）。兩者預設 0，行為與舊版逐位元相同（與 HEAD 版跑同一組 rollout，SHA256 相同）。需要 `--payload-max > 0`。
- #73（line6_pl20_k2_h4）已用 `gpujob kill 73` 取消，未跑過。
- 新排隊（都是 line6:T1 + line6:T2 各 512 環境、k=2、H=4、1.5e8 步、time-limit 300 分鐘、--seed 7；VRAM 估計約 4GB；記憶體上限 12G）：
  - #75：results/p3/line6_pl20off_ev_k2_h4（`--payload-max 0.20 --payload-offset-max 1 --payload-event-prob 0.3`）
  - #76：results/p3/line6_nom_k2_h4（無 payload，標稱基線）
- 評估指令（CPU，完成後；#76 用 E1 時不適用，只評標稱即可）：
  E1：CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.rl_eval results/p3/line6_pl20off_ev_k2_h4/final.pt --configs line6 --tasks T2 POS_STEP --payload payload15 --out results/p3/line6_pl20off_ev_k2_h4/eval_E1.json
  E2：… --holdout --out …/eval_E2.json
  E3（偏離中心固定點：15%、argmax 掛點、offset (+0.0333, −0.0333, −0.0215) m；對照 B1-QP、B1-QP-yawlim、estim-yawlim(snap=False)、informed-offset-yawlim）：… --tasks T2 POS_STEP --payload E3 --out …/eval_E3.json
  RQ-D 事件（attach／drop × 中心／偏離中心，任務 T2 POS_STEP HOVER；與 results/p2/rqd_classical_raw.json 逐 episode 成對）：… --tasks T2 POS_STEP HOVER --event --out …/eval_event.json
  標稱基線：CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python -m scripts.rl_eval results/p3/line6_nom_k2_h4/final.pt --configs line6 --tasks T2 POS_STEP（可再加 --payload payload15 / E3 看退化）。
- 注意：rl_eval 的古典比較會寫 results/p3/rl_eval_cache.json，其 _meta 含 n_eps，用非 100 的 --n-eps 會清掉快取；冒煙測試請先備份。

### train4_mix_k1_h4 評估結果（#55 已跑完，3e8 步，6812 秒；eval：results/p3/train4_mix_k1_h4/eval_final.json，每格 100 個 episode）
比值 = RL 分數 / B1 分數（括號：RL 成功率）。**驗收（每格 ≥ 0.90）未通過：12 格中 4 格 < 0.90（line2、line3、square4 的 T2，line3 的 ATT_STEP）。**
| 構型 | ATT_STEP | POS_STEP | T2 |
|---|---|---|---|
| single | 0.986 (100%) | 1.007 (100%) | 0.937 (87%) |
| line2 | 0.991 (100%) | 0.997 (100%) | 0.816 (77%) |
| line3 | 0.871 (91%) | 0.988 (100%) | 0.693 (63%) |
| square4 | 0.930 (100%) | 0.958 (100%) | 0.806 (77%) |
- 差距集中在 T2（成功率只有 63–87%，B1 是 99%）與 line3 的 ATT_STEP；ATT_STEP/POS_STEP 在 single、line2、square4 都達標。對 max(B1,B1-QP) 更差（T2：0.90/0.77/0.67/0.73）。
- 原因判斷（未驗證）：(1) 多構型 + k=1 H=4 比單機 k=0 H=1（T2 成功率 98%）學得慢，3e8 步時 step_reward 0.29 仍在上升，可能只是步數不足；(2) T2 的失敗是墜機/飛太遠（成功率低），不是追蹤慢；(3) 構型越大越差，與 line3 比 line2 更差一致。
- 下一步：從 final.pt 續訓（rl_train 目前沒有 --init，需加）或增加到 1e9 步；也可看 train.csv 分構型的 success_rate 曲線確認是否仍在爬升；再用 p3_diag 看 line3 T2 的終止原因。
- 指揮官補充（2026-10-30 20:50）：訓練中 T2 成功率在 7.5e7 步時已是 0.63–0.80，到 3e8 步仍是 0.70–0.86（train.csv 分構型欄），**是平台，不是步數不足**，原因 (1) 不成立。對照：單機 k=0 H=1 T1+T2 混合 T2 成功率 98%。
- 因此 #75/#76（同為 k=2 H=4 配方，各最多 5 小時）先取消，改排：#81 abl_single_mix_k1_h4（single:T1+T2、k=1、H=4、1.5e8 步）分離「k/H 的影響」與「多構型共用的影響」；之後 #82 line6_nom_k2_h4（原 #76）、#83 line6_pl20off_ev_k2_h4（原 #75），參數不變。
  - #81 T2 成功率 ≈ 98% → k/H 無害，問題在多構型共用；#81 明顯低 → 問題在 k/H（歷史輸入或 GNN），先修配方再跑 #82/#83。

### #81 abl_single_mix_k1_h4 結果（驗證 seed，offset 100，每格 100 ep；eval_val.json）
| 任務 | RL 成功率 / 分數 | B1 | 比值 vs max(B1,QP) |
|---|---|---|---|
| ATT_STEP | 1.00 / 0.807 | 0.772 | 1.046 |
| POS_STEP | 1.00 / 0.681 | 0.646 | 1.054 |
| T2 | 1.00 / 0.457 | 0.415 | 1.099 |
- 判讀：k=1、H=4 在單機上沒有傷害（和 k=0 H=1 的 1.04–1.09 相當），所以 train4 的 T2 平台來自**多構型共用訓練**，不是 k/H。單機配方通過 prereg 7.3(a)（每格 ≥ 0.90）。
- #82/#83 都是單一構型（line6），沒有多構型共用的問題，照原計畫跑。P3a（多構型）的修正，例如每構型的 reward 正規化或加大批次，列為之後的工作，不擋 P4（依 prereg 7.3）。

### #82 line6_nom_k2_h4 結果（驗證 seed offset 100；eval_val.json）
| 任務 | RL 成功率 / 分數 | B1 | 比值 vs max(B1,QP) |
|---|---|---|---|
| ATT_STEP | 0.59 / 0.438 | 0.721 | 0.606 |
| POS_STEP | 0.71 / 0.451 | 0.643 | 0.701 |
| T2 | 0.54 / 0.258 | 0.328（B1-QP 0.347） | 0.745 |
- 訓練成功率逐步下降（T2 從 0.61 降到 0.45，T1 從 0.55 降到 0.36，3.7e7 → 1.5e8 步）。**line6 在標稱條件下就學不起來**，所以 payload 先導 #83 沒有意義，已用 gpujob kill 83 取消。
- 下一步：先做 CPU 診斷，找出原因（政策崩壞？value？探索雜訊是否隨 N 累加？），之後再花 prereg 第 10 節剩下的 2 次配方修改預算。

### 配方修改第 1 次：動作均值超界懲罰（#85 line6_nom_mb_k2_h4，--mean-bound-coef 0.01，4e7 步）
- 對照組：line6 T2 的飽和比例，B1 0.149、B1-QP 0.170，原 RL 0.490，所以 RL 的飽和約是 B1 的 3 倍，是 RL 特有的。
- 加了懲罰之後：飽和比例 0.157，|μ|>1 的比例從 0.48 降到 0.15。訓練末 20 輪的 T2 成功率 0.593，原 run 在同步數是 0.453；但 T1 成功率從 0.523 降到 0.272。
- 驗證 seed 上的分數比值 vs B1：T2 0.929、POS 0.892、ATT 0.491。原 run 是 0.79、0.70、0.61。
- 混淆因素：4e7 步的 run，學習率在 4e7 步內就退火到 0；原 run 在 4e7 步時學習率還有約 73%，兩者不是在相同條件下比較。
- 判讀：飽和確實修好了（降到和 B1 同一水準），但整體沒有一致變好（ATT 和 T1 變差）。飽和比較像是**症狀**，不是主因。依預先寫好的規則，假說「沒有被否定」，但證據不足以說成立。
- results/p3/line6_pl20off_ev_k2_h4/ 是 #83 被取消前留下的半成品，不使用。

### 配方修改第 2 次（最後一次）：整隊一筆樣本（#122 line6_nom_team_k2_h4，--team-sample --mean-bound-coef 0.01 --n-minibatches 8，1.5e8 步）
- 訓練成功率（前後 20 輪平均）T2 / T1：3.7e7 步 0.45 / 0.49；7.5e7 步 0.37 / 0.60；1.5e8 步 0.37 / 0.68。原 run 是 T2 0.61→0.45、T1 0.55→0.36。
- 驗證 seed（offset 100）分數比值 vs B1：ATT 0.759（成功率 0.78）、POS 0.405（0.43）、T2 0.626（0.42）。原 run 是 0.61 / 0.70 / 0.79。
- 判讀：ATT 和 T1 改善，但 POS 和 T2 明顯變差，**沒有達到 prereg 7.3(b)**（line6 須通過 go/no-go）。配方修改預算（2 次）已用完。
- 依 prereg 7.3：P4 的 RL 主實驗不開，RL 照負面結果報告。報告要附上本節以及前面各節的診斷。

## 探索性殘差 RL（prereg §13，不作判定）— res_line6_off_k2_h4（gpujob #131）
- 設定：u = clip(u_estim-yawlim + 0.2·δ)；line6:T2 + line6:POS_STEP；payload-max 0.20、offset-max 1；120 分鐘上限，實跑 32.7M 步（1000/1831 iter）。訓練成功率 0.93–0.99。
- 驗證 seed（offset 100），全部 line6：

| 條件 | 任務 | 殘差 RL | estim-yawlim | informed（特權） | 成對差 RL−estim [95% CI] |
|---|---|---|---|---|---|
| 標稱 | ATT / POS / T2 | 0.723 / 0.641 / 0.416 | （未列；B1 0.721 / 0.643 / 0.328） | — | — |
| E1 payload15 | ATT / POS / T2 | 0.581 / 0.635 / 0.351 | 0.588 / 0.636 / 0.356 | 0.623 / 0.638 / 0.251（B1-QP-informed） | T2 −0.005 [−0.010, −0.001] |
| E3 偏心 | ATT / POS / T2 | 0.353 / 0.554 / 0.225 | 0.354 / 0.565 / 0.239 | 0.358 / 0.604 / 0.245 | POS −0.010 [−0.013, −0.008]；T2 −0.014 [−0.022, −0.008] |

- 解讀：殘差沒有縮小 E3 POS_STEP 的缺口（0.565→0.604），反而略低於 estim-yawlim（差 1–2%，CI 不含 0）。E1 與 estim 幾乎相同。標稱 T2 為 B1 的 1.27 倍，但這輪評估沒有列出標稱下的 estim-yawlim，不能歸功於殘差（yawlim 本身就會改善 T2）。
- 結論（探索性）：在這個預算下，殘差 RL 學到的是「接近 0、略有傷害」的修正，沒有證據顯示它能補上古典法的剩餘缺口。不再追加 RL 實驗。

## seed 8 補跑（prereg §14(b)，探索性，不作判定）— gpujob #254、#255
- 同參數、只改訓練 seed 為 8（`res_line6_off_k2_h4_s8`、`line6_nom_mb_k2_h4_s8`）。#254 實跑 7203 秒（120 分鐘上限，1001 iter，約 3.28e7 步），#255 實跑約 1900 秒。評估：驗證 seed（offset 100）、line6、每格 100 episode、final.pt。

| run | 條件 | ATT / POS / T2 分數 | 成對差 RL−estim-yawlim [95% CI] |
|---|---|---|---|
| 殘差 seed 7 | 標稱 | 0.723 / 0.641 / 0.416 | — |
| 殘差 seed 8 | 標稱 | 0.722 / 0.640 / 0.414 | — |
| 殘差 seed 7 | E1 payload15 | 0.581 / 0.635 / 0.351 | T2 −0.005 [−0.010, −0.001] |
| 殘差 seed 8 | E1 payload15 | 0.580 / 0.634 / 0.350 | T2 −0.006 [−0.011, −0.003] |
| 殘差 seed 7 | E3 偏心 | 0.353 / 0.554 / 0.225 | POS −0.010 [−0.013, −0.008] |
| 殘差 seed 8 | E3 偏心 | 0.353 / 0.555 / 0.224 | POS −0.010 [−0.011, −0.009] |

| run | 標稱 ATT / POS / T2 比值 vs B1（成功率） |
|---|---|
| mean-bound seed 7（#85） | 0.491 (0.48) / 0.892 (0.92) / 0.929 (0.67) |
| mean-bound seed 8（#255） | 0.634 (0.64) / 0.707 (0.73) / 0.661 (0.46) |

- 殘差 RL：兩個 seed 各格分數差 ≤ 0.002（成功率差 ≤ 0.01），都略低於 estim-yawlim、遠低於 informed（E3 POS 0.604）。「殘差學到接近 0、略有傷害的修正」這個觀察在第二個 seed 上也看到（每個 seed 各只有一次訓練）。
- mean-bound 從頭訓練：seed 8 三格都 < 0.90，仍未過 go/no-go；各任務的高低和 seed 7 不同（seed 7 的 T2 0.929 在 seed 8 掉到 0.661），表示從頭訓練的跨 seed 變異很大，seed 7 的單一數字不能當成穩定值。
- 結論不變：RL 是負面結果。兩個 seed 仍是探針，不做跨 seed 推論。
