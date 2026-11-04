# 預先登記（prereg）：P4 主實驗的判定規則

狀態：v1.0（凍結版，2026-11-01；凍結說明見 §14）。P4 主實驗依 7.3 未開，標為「不適用」的 `[待填]` 保留原文規則、不刪除；凍結時間點 = 任何 RL 策略在 confirmatory seed 上被評估之前（實際上沒有任何 RL 策略在 confirmatory seed 上評估過）。凍結後的任何修改必須在「修訂紀錄」一節逐條揭露。
撰寫日期：2026-10-30。依據：`PLAN.md` v7.1、`PROGRESS.md`、`results/p3/STATUS.md`、`results/p2/` 各報告。
**揭露（摘要）**：E3 這一格與 snap=False 主設定，都是**看過古典結果之後才選的**（選擇時沒有任何 RL 評估參與，細節見 4、5.b），不是事前決定；報告必須原樣揭露。總結論只依「E3 的 POS_STEP 三格經 Holm 校正後的結果」與「E1 三格各自的等價結果」寫成（3.2、5.d）。
**揭露（v0.4 改 E3 主任務）**：(1) v0.3 的 E3 主任務原為 T2，是根據 T2+POS 合併值選的；分任務重算後發現 T2 的 estim→informed 缺口 ≤ 0.007，幾乎沒有缺口，缺口幾乎全在 POS_STEP，所以 v0.4 改用規則選任務（5.b），結果為 POS_STEP。(2) 這次改動在任何 RL 策略於 E3 上評估之前完成，只依據古典結果；探索 seed（offset 0）已足以決定，confirmatory seed 上的古典結果模式也相同。(3) 凍結（git tag）之後的任何改動，一律列為「偏離預先登記」，並在報告中逐條列出。
標記約定：`[推論/提案]` = 文件撰寫者的提案，非使用者已定案；`[待填]` = 凍結前必須填入的值（每個 `[待填]` 由指揮官在 7.4 順序表指定的步驟填入）；`[出處待確認]` = 找不到單一檔案出處；`[待實作]` = 凍結前必須完成的程式與測試。v1.0 起：只為 P4 服務的 `[待填]` 改標「不適用：P4 未開（7.3），見 §14」；古典端 confirmatory 數字改標「凍結後執行的確認性報告」。

---

## 1. 研究問題

分析型結論。模組化對接多旋翼（一個主體、一個大腦）的控制有三個瓶頸，各有古典解：

| 瓶頸 | 古典解 | 實作 | 出處 |
|---|---|---|---|
| 形狀資訊（k 跳局部資訊） | QP 分配 | B1-QP / B2-QP | `results/p2/g1_v2.md` |
| 飽和協調 | yaw 參考限速 | B1-QP-yawlim | `results/p2/g1_coord.md` 事實陳述 2、5；`g1_yawcost.md` |
| 未知 payload | 質量/重心線上估計 | B1-QP-estim | `results/p2/g2_estim.md` |

RL 在此只是探針。問題：**在古典解全開仍有缺口的格子，學到的控制器（actor 看不到 payload）能否補回缺口？** 另問：RL 在古典主判斷格能否追平估計式古典法，以及在估計器假設失效（中途事件）時誰比較穩。

所有結論的措辭限定為「**相對於狀態回授的古典法**」（見 2.1），不寫成「相對於古典法」一般。

負面結果照實報告：RL 落後、不確定、或古典法已補平，都是有效結論，不換判斷格、不事後挑對手。

---

## 2. 措辭限制

- 「補回」只指**本 env 的 payload 模型內**：payload 是掛在模組中心（或固定偏移）的質點、非質點與柔性不在其內。出處：`PLAN.md` v7.1 措辭界線與已知限制。
- 估計器與 env 共用 payload 模型（質點），MuJoCo 重現只能驗積分器，治不了這個循環；循環由偏離中心與 snap=False 條件檢驗（`PLAN.md` v7.1 次要／附錄）。
- line8 payload15 T2 的剩餘落差（estim+yawlim 0.253 對標稱 0.378；informed+yawlim 0.255）比資訊缺口大，是物理權限問題，不寫成資訊缺口。出處：`results/p2/g2_estim.md` 第 44 行、`g1_coord.md`。（這些是 snap=True 數字，僅作背景敘述，不是凍結值。）
- 古典對手「逐格挑最強」的部分（g1_v2、g1_coord、g2 的 C）對古典法有利，會低估 RL 的相對表現。本文件的主比較對象已在第 4 節固定，不再逐格挑。

### 2.1 雙方能拿到的資訊（資訊不對稱，正文必讀）

| | RL（actor） | RL（critic，僅訓練時用） | 古典 B1 系列（含 estim） |
|---|---|---|---|
| payload 質量／位置 | 看不到 | 特權欄位 PRIV_DIM=4（`quadsim/policy.py:31`）；critic 不在部署時使用 | 看不到（informed 對照組除外，那是特權參考） |
| 位置、速度、姿態 | 真實值（機體座標位置誤差、速度、旋轉矩陣前兩欄），無雜訊（`quadsim/env.py:334-352` `observe`） | 同 actor | **真實狀態**：外迴路直接讀 `env.state()` 的 pos、vel、q、w，無雜訊（`quadsim/controllers.py:72-73`；`quadsim/env.py:488-490`） |
| 角速度 | 帶雜訊陀螺儀（gyro_noise=0.01 rad/s，`env.py:92,343`） | 同 actor | 外迴路用**真實 w**（`controllers.py:72`）；僅 estim 的估計器另用帶雜訊陀螺儀與加速度計（`controllers.py:810-811`） |
| 加速度 | 帶雜訊加速度計（accel_noise=0.05，`env.py:93,344`） | 同 actor | estim 估計器用帶雜訊加速度計（`controllers.py:811`）；外迴路不用 |
| 鄰居／歷史 | k 跳鄰居旗標、H 步歷史（`rl_train.py` 參數；env 的 neighbor_flags 與 u_prev 在 obs 中） | 同 actor | 構型形狀：B1 系列的 leader 知道整架標稱形狀與總體角加速度權限（`controllers.py:65-71` docstring 自述「對古典法有利」） |

結論：兩邊的位置／速度／姿態都是真實值；差別在 (1) 古典外迴路用無雜訊角速度，RL 用有雜訊的；(2) 古典 leader 知道整架標稱形狀，RL 只有 k 跳資訊；(3) RL 要從歷史推斷 payload，古典 estim 用顯式 RLS 估計。因此這個比較**對古典法在角速度與形狀資訊上有利**，對 RL 在「顯式模型結構」上不利；結論只能寫成「相對於狀態回授的古典法」。**A3（古典法外迴路也吃有雜訊角速度）是必報項目**，見 6.3，不在附錄。

- 古典法位置／姿態回授用真實狀態；estim 的質量／重心估計只用帶雜訊的陀螺儀、加速度計與自己的指令；RL actor 用帶雜訊的陀螺儀與加速度計（其餘 obs 為真實值）。

---

## 3. 分數、缺口、seed 分組與統計

- **分數** = success × max(0, 1 − e_rms / e_ref)。e_ref：ATT_STEP 1.0、POS_STEP 1.5、T2 1.5。出處：`scripts/g1_gate.py:45-47,95`。
- **缺口（pp）** = 100 × (1 − 方法分數 / 分母分數)。分母於各判斷格明載。
- **每格 episode 數**：100（`scripts/rl_eval.py:269` 預設 `--n-eps 100`）。
- **差（RL 對 estim）** d_s = 100 × (RL_s − estim) / estim，RL_s 是第 s 個訓練 seed 的策略在該格 100 個 episode 的平均分數；estim 為同一 100 個 episode（逐 episode 成對）的平均分數（確定性古典法，沒有訓練 seed）。

### 3.1 seed 分組（三組不得重疊，清單寫死）

環境 seed 以 `SEED[task]`（ATT_STEP 0、POS_STEP 1、T2 2，`g1_gate.py:45`）加位移得到；HOVER 為 SEED_EV 的 3（`rqd_classical.py:47`）。`rl_eval.py` 目前直接用 `SEED[task]`，**[已完成，commit 5b4fe04]** 加 `--seed-offset INT`（比照 `g2_estim.run` 的 `seed_off`）：驗證用 `--seed-offset 100`、confirmatory 用 `--seed-offset 900000`、探索用 `0`（預設）。

| 組 | 位移 | 環境 seed（ATT_STEP / POS_STEP / T2 / HOVER） | 用途 |
|---|---|---|---|
| **驗證** | +100 | 100 / 101 / 102 / 103 | RL 的 go/no-go（7.1）與 7.2 診斷（H、PRIV_DIM）、3.4 的先導 SD。**只能用這組**。古典端的驗證 seed 本來就是 100+SEED[task]（`g2_estim.py:32` VAL_OFF=100；`rqd_classical.py:46` VAL_SEED_OFF=100），RL 與其同性質同值。 |
| **探索** | +0 | 0 / 1 / 2 / 3 | 舊評估 seed。`p3d_factorial`、`p3d_resolve`、`rqd_classical` 已在其上看過結果，P3a 評估與 STATUS 的 RL 數字也在其上，因此**只當探索**，不用於任何判定。 |
| **confirmatory** | +900000 | 900000 / 900001 / 900002 / 900003 | 第 5 節所有主判斷格**只用這組**判定。古典基準（estim、informed、estim-reset 等）也在這組上重算。 |

**快取與 seed 規則（機械執行）**：
- 評估一律用 `rl_eval.py --seed-offset`，不得手改 `SEED` 常數。
- `rl_eval` 的古典端快取 key 必須含 seed（實際環境 seed 或 offset），輸出 meta 也必須記錄 seed offset。
- `--seed-offset` ≠ 0 時，**不得沿用** `results/p2/g2_raw.json`（它是探索 seed 的快取）；offset ≠ 0 的古典基準一律重算，並寫到另外的檔案。
- 凍結前 `tests/test_rl_eval_seed.py` 必須通過（至少驗：不同 offset 得到不同環境 seed；快取 key 含 seed；offset ≠ 0 不讀 g2_raw.json；meta 記錄 offset）。測試沒過不得凍結。

- 其他環境 seed 之外的亂數：`g2_estim.py` 的 `TUNE_SEED=20261030`（numpy rng，調參抽樣）、`HOLD_SEED=12345`（E2 掛點抽樣 rng）不是環境 seed，不在此表內。`tune_classical.py:25` 的 `TUNE_SEED=1000`（TRAIN 構型的古典調參環境 seed）與訓練 seed 1 的環境 seed 1000 同值，但那是古典 TRAIN 調參，與 P4 無關，僅記錄。
- 每個 episode 的初始條件、事件時刻、掛點都只由 (env seed, 環境槽, episode 編號) 決定（`env.py:396,460-464`），所以不同的 env seed 整數就是不同的資料；三組之間只要整數不同即不重疊。

**訓練 seed**（`rl_train.py:84,88`：環境 seed = `args.seed * 1000 + i`，i = 該 run 的第 i 個 (構型, 任務) 組，i < 1000）：

| 項目 | 訓練 seed `--seed` | 環境 seed 範圍（= seed×1000 + i） |
|---|---|---|
| 主要 8 個 | 101, 102, 103, 104, 105, 106, 107, 108 | 101000+i … 108000+i |
| 延長用（3.5 的觸發條件成立才使用，事先列出） | 109, 110, 111, 112 | 109000+i … 112000+i |

檢查：i 最多為 (構型, 任務) 組數減 1，P4 配方每個 run ≤ 8 組，遠小於 1000，所以每個訓練 seed 的環境 seed 落在 [s×1000, s×1000+999]。7.2 診斷另用訓練 seed 201、202（環境 seed 201000+i、202000+i），它們不屬於 101–112，也不與其他組相交。與其他組：驗證 100–103 < 101000；探索 0–3 < 101000；confirmatory 900000–900003 > 112999（最大訓練環境 seed ≤ 112999）。三者不相交，且與先導用的 seed 7（7000/7001）、預設 seed 0（0–7）也不相交。`[不適用：P4 未開]` 凍結前在 `rl_train.py` 或單元測試加一個斷言：訓練 seed ∈ {101..112} 且環境 seed 不在 {0..3, 100..103, 900000..900003}；診斷 seed 201、202 另行允許，但同樣檢查環境 seed 不在該集合內。

### 3.2 統計方法（主方法 + 敏感度）

- **單位**：訓練 seed（n = 3.4 定死的 8 或 12）。每個 d_s 是一個觀測。**所有訓練 seed 共用同一組評估 episode**（同一個 confirmatory env seed、同一組 100 個 episode，與 estim 逐 episode 成對），所以 t 區間**只涵蓋訓練 seed 的變異，不涵蓋評估 episode 的抽樣變異**。5.a、5.b 的結論因此一律寫成「**條件於固定評估集**」。
- **主方法（寫死）**：以 seed 為單位的 **t 區間**，自由度 n−1，對 d_s 的均值。兩種區間水準：
  - 等價檢定（TOST，α=0.05 每側）＝ 90% 雙側 t 區間（t_{0.95,7}=1.895；n=12 時 t_{0.95,11}=1.796）；
  - 優越性檢定（單邊，α=0.05）＝ 單邊下界 = 均值 − t_{0.95,n−1}·SD/√n，與 90% 雙側區間下界相同。
  - 另報 95% 雙側 t 區間（t_{0.975,7}=2.365）作描述，不用於判定。
- **敏感度分析**：以 seed 為單位的 **BCa bootstrap**（N_boot=10000，同樣的 α）。**BCa 在 n=8 下同樣不可靠**（加速度與偏誤修正都由 8 個點估計），僅作參考。主方法與 BCa 的判定結果不一致時，兩個都報、判定以主方法為準，並在報告中標明該格「對方法敏感」。
- 舊版的「n=8 百分位 bootstrap」取消，因為 n=8 時百分位區間偏窄。
- **兩層 bootstrap（seed 層 + episode 層）（不適用：P4 未開，見 §14；以下為原規則）**：現有 `rl_eval.py::_boot_diff`（`rl_eval.py:142`）只做 episode 層（nb=1000）。凍結前必須實作「先抽 seed、再在抽到的 seed 內對共用的 episode 索引重抽」，並寫測試（至少：(a) 合成資料上覆蓋率合理；(b) 退化到單一 seed 時等於 episode bootstrap；(c) 固定隨機種子下可重現）。實作完成後，其結果**與 t 區間並列報告**（因為它額外涵蓋 episode 抽樣變異，是處理固定評估集問題的補充）；**判定仍以 t 區間為準**。**不實作完成不得凍結**。
- **多重比較（v0.3 寫死）**：
  - **E3 主檢定**：只用 POS_STEP（任務由 5.b 的規則選出），共 3 格（line6、line7、line8），單邊優越性檢定（3.3 乙），三格合為**一個家族**，用 **Holm 校正**，家族 α = 0.05（依各格單邊 p 值或等價地依 t 統計量排序，逐步比較 0.05/3、0.05/2、0.05/1）。
  - E3 的 T2 單獨與 T2+POS（串接）為**次要**：只報告差與 CI，不判定，不入家族。
  - **E1 的 3 格**是等價檢定：各格分開報告，**不合併、不校正**。
  - **RQ-D 的 48 格改為描述性**：只報「落在等價界 ±B% 內的格數比例」，以及每格的估計值與 CI；**不逐格宣稱**追平／超過／落後。
  - **總結論只依 E3 校正後的結果與 E1 的等價結果寫**（5.d）。其餘一律是描述。
- **所有訓練 seed 都要報**：訓練發散或當掉的 seed 不得剔除，所有格的分數記 0（成功率 0 → 分數 0），當作失敗計入均值與 SD。**checkpoint 一律用 `final.pt`**，不得挑選（不用 best、不用 latest、不用中途 checkpoint）。系統性當機（非發散）可用同 seed、同配方重跑一次；重跑仍失敗就記 0，不換 seed。

### 3.3 判定規則

**(甲) 等價檢定（用於 E1；RQ-D 只借用區間做描述，見 3.2）**：界值 ±B%（預設 ±10%，3.4 可放寬，上限 ±15%），相對 estim。令 [L, U] = 90% 雙側 t 區間：
- [L, U] ⊂ (−B, +B) ＝「追平」（等價）
- L > +B ＝「超過」
- U < −B ＝「落後」
- 其餘 ＝「不確定」

**(乙) 單邊優越性檢定（用於 E3 的 POS_STEP 三格）**：H1：RL > estim-yawlim（snap=False）。每格單邊 t 檢定（自由度 n−1），三格為一個家族，經 Holm 校正後家族 α=0.05：某格被拒絕 H0 ＝「顯著優於」。等價地，未校正時單邊下界 L > 0 是必要條件。解讀見 5.b。

出處：甲沿用 `PLAN.md` v7.1 RQ-R 判準但把 95% CI 改為 TOST 90% 區間（TOST 的標準做法）；乙為 v0.2 新增（理由見 5.b）。

### 3.4 功效分析與 n（一次，P4 開跑前定死）

- **先導**：用 3.1 的前 3 個 P4 訓練 seed（101–103，凍結配方，final.pt，不挑）在**驗證 seed**（不是 confirmatory）上評估 E1 line6 T2，得到先導 SD_rel = SD(d_s)（單位 %，樣本標準差，n=3）。**先導 SD 只有約 3 個點，是粗估**（3 點 SD 本身的不確定性很大），規則仍照機械式執行，不另做修正，但報告須註明這一點。這 3 個策略仍然是正式 seed，不剔除。
- **n 的規則（機械）**：hw8 = 1.895 × SD_rel / √8 = 0.670 × SD_rel。
  - 若 hw8 ≤ 8%（= 0.8 × 10%）→ n = 8，B = 10%；
  - 若 hw8 > 8% → n = 12（使用延長 seed 109–112，**在看到任何 confirmatory 結果之前決定**），hw12 = 1.796 × SD_rel / √12 = 0.5185 × SD_rel；
  - 在選定的 n 下：若 hw_n ≤ 8% → B = 10%；若 8% < hw_n ≤ 12%（= 0.8 × 15%）→ B = 15%；若 hw_n > 12% → B = 15% 且報告預先註明「功效不足，預期多數格為不確定」，**不再放寬、不再加 seed**。
- n 與 B 只決定一次，只能依 SD_rel，**不得看 confirmatory 的均值、區間或結論**，也不得在 confirmatory 評估後改。決定的日期、SD_rel、n、B 寫入此處與 `PROGRESS.md`：SD_rel：不適用：P4 未開（7.3），見 §14；n：不適用：P4 未開（7.3），見 §14；B：不適用：P4 未開（7.3），見 §14；日期：不適用：P4 未開（7.3），見 §14（原規則保留不刪；P4 未開，先導 SD 從未執行，n 與 B 不決定）。
- B=15% 的依據：對齊 Hopper 驗收界 ±15%（`PROGRESS.md`「使用者決定」Hopper 條）。
- 此節填完之後才凍結。E3 的優越性檢定沒有界值，只用 n（Holm 校正不改變 n 的規則）。

### 3.5 延長到 12 的限制

- 延長**只能**由 3.4 的先導 SD 規則觸發，在 confirmatory 評估前決定，一次定死；沒有「看 8 seed 結果再決定要不要加」的路徑。舊版「CI 半寬 > 界值 → 序列延長」取消，因此也不再有序列檢定的 α 膨脹問題。
- 不得延長超過 12；不得改變 3.1 清單。

---

## 4. 比較對象（固定，不逐格挑）

| 名稱 | 設定 | 角色 |
|---|---|---|
| **主古典對手** B1-QP-estim-yawlim | `snap=False`（`nosnap=True`），超參數取 `results/p2/g2_estim_tune.json` 第 8 組（驗證 seed 100+、20 組搜尋），yawlim 參數 rate_c=0.556、yaw_ratio=0.415（TRAIN 上 20 組搜尋，`PLAN.md` v7.1 已知限制） | RL 的判定基準（分母） |
| 輔助對手 B1-QP-estim（無 yawlim） | 同上 | 並列報告，不當分母 |
| B1-QP-estim-reset-yawlim | CUSUM 重置版，偵測超參數 {k:1, h:12, chan:mc, refr:30}（`results/p2/rqd_tune.json`、`rqd_classical.md` 標頭） | 僅用於事件格（5.c） |
| informed 參考（只當參考上限，不當缺口分母，除非另註明） | B1-QP-informed-yawlim（E1）；**B1-QP-informed-offset-yawlim**（E3，知道偏移）；**B1-QP-informed-event-yawlim**（事件格，知道事件） | 特權參考 |
| 其他古典 | B1-QP、B1-QP-yawlim、B3 INDI、B3+adapt | 報告對照 |

**snap=True 的處理**：把重心估計吸附到最近模組中心。在 payload 掛模組中心的條件下與 snap=False 幾乎相同（payload15 line8 T2 0.253 對 0.255；`results/p2/p3d_factorial.md` 事實陳述 6），但偏離中心時全面失效（line6/line8 T2 的 QP+yawlim+estim 0.000 對 snap=False 的 0.197／0.131；同一事實陳述 6）。snap=True 列入報告「什麼沒成功」一節。

**揭露（選擇過程）**：snap=False 作為主設定，是在**看過偏離中心診斷結果之後**才選的（`p3d_factorial.md`、`p3d_resolve.md`、`rqd_classical.md` 顯示 snap=True 在偏離中心時失效），不是事前決定。這個選擇使用的資料全是古典端結果，沒有任何 RL 評估參與；但它仍是看過資料後的選擇，報告必須原樣揭露，並同時給出 snap=True 的主判斷格數字當對照。

不確定項 `[出處待確認]`：`PLAN.md` v7.1 RQ-R 寫「B1-QP-estim（含 / 不含 yawlim，設定在驗證集定死）」，但驗證集定案紀錄中找不到；使用者指示以 snap=False + yawlim 為主設定，本文件照此固定。

---

## 5. 主判斷格

### 5.0 主判斷格清單（固定，不得刪格）

| 編號 | 格 | 判定 |
|---|---|---|
| E1 | line6 T2、line7 T2、line8 T2（payload15） | 等價檢定（3.3 甲） |
| E3 | **主檢定：line6 POS_STEP、line7 POS_STEP、line8 POS_STEP（3 格；任務由 5.b 規則選出）**；T2 單獨與 T2+POS（line6/7/8，T2 與 POS_STEP 串接）為**次要，只報告、不判定** | 單邊優越性檢定（3.3 乙），3 格一個家族、Holm 校正、α=0.05；補回比例（描述） |
| RQ-D | `rqd_classical.py` 的全部格：構型 {line6, line7, line8, grid10} × 任務 {T2, POS_STEP, HOVER} × 條件 {attach-center, attach-offcenter, drop-center, drop-offcenter}（`rqd_classical.py:42-44`），共 48 格，指標 = 整段分數 | **描述性**：只報「落在 ±B% 內的格數比例」與每格估計值＋CI，不逐格宣稱；基準 = estim-reset-yawlim |
| E2 | 留出集 | **次要格，只報告、不判定** |

- 清單內的格一個都不得刪。某格若無法計算（例如古典基準缺、某構型事件不支援），報告該格為「**無法計算**」並寫原因，不得靜默略過，也不得改成「只報 line6/8」。
- 次要（只報差與 CI，不判定）：E3 的 T2 單獨與 T2+POS、E1 的 POS_STEP 單獨、RQ-D 的事件後峰值／恢復時間／成功率、E2、6.1 的退化比。
- line7 於 `p3d_factorial.md` 事實陳述 2 標示「未逐一核對」，其 E3 缺口見 5.b 的 confirmatory 表（line7 POS_STEP estim 0.498／informed 0.559，T2 0.182／0.181，出處 `PROGRESS.md` 第 531 行）；不因此刪格。

### 5.a E1：payload15 line6–8 T2

- **條件**：payload 15%、掛在 argmax(x) 最遠端模組、無偏移、t=0 即存在（`scripts/g2_mismatch.py:66`，經 `rl_eval.py --payload payload15`）。
- **基準數字**（B1-QP-estim-yawlim，snap=False，confirmatory seed 重算）：line6、line7、line8 的 estim-yawlim、B1-QP-informed-yawlim 與無 yawlim 版 estim：凍結後於 confirmatory seed（offset 900000）執行，結果寫入 `results/p2/classical_confirm.md`；不改任何規則或比較對象（見 §14）。**判讀規則**：古典方法的結論以 confirmatory 為準；若與探索 seed 的結論方向不同，兩者都報、以 confirmatory 為主。舊文件的 0.307／0.285／0.253（estim-yawlim）與 0.308／0.291／0.255（informed-yawlim）是 **snap=True、探索 seed** 的數字（`g2_estim.md` 第 38–44 行），**不是凍結值**，僅作背景。
- **預期**：古典 estim 對 informed 在探索 seed 上補回比例 0.99/0.96/0.98，所以 RL 可上升的空間 ≈ 0；本格的科學意義 = 「RL 能否追平估計式古典法」，不是「補缺口」。
- **判定**：3.3 甲，分母 = 主古典對手分數。三個構型各給一個結論，不合併、不做多重比較校正；構型間結論不一致時三個都報。結論措辭一律寫成「**條件於固定評估集**（confirmatory env seed 的同一組 100 episode），區間只涵蓋訓練 seed 變異」，並並列報兩層 bootstrap 區間（3.2），判定以 t 區間為準。

### 5.b E3：最遠端、偏離中心、t=0 就有 payload（RL 探針主判斷格）

- **條件**：payload 15%、掛 argmax(x) 最遠端模組、offset = (+0.0333, −0.0333, −0.0215) m（= 0.5·s_max 規則，s_max=1.4357，OFFSET_SCALE=0.718，`p3d_factorial.md` 事實陳述 7）、t=0。實作 `scripts/rl_eval.py --payload E3`，`p3d_factorial.OFFSET`。
- **揭露（選格過程）**：E3 這一格是在**看過古典缺口分解（`p3d_factorial.md`、`p3d_resolve.md`）之後**才選的：選格發生在任何 RL 評估之前，只依據古典結果（古典全開法在這一格仍落後 informed-offset）。這是看過資料後的選擇，報告必須原樣揭露。
- **為何用單邊優越性檢定而非等價檢定**：全開古典法（QP+yawlim+estim snap=False，不論有無重置）在這一格仍落後知道偏移的 informed 參考；探索 seed 上合併缺口（line6/8、grid10 × T2/POS_STEP）約 8.4 pp [6.7, 10.1]，line8 T2+POS 16.4 pp [11.9, 21.2]；估計器版本貢獻 0 pp（`p3d_resolve.md` 事實陳述 5）；缺口分析另見 `p3d_factorial.md` 事實陳述 5（snap=False 全開缺口 4.6/5.6/10.9/11.3/7.6/7.9 pp，line6/7/8/grid10/line3/square4）。**缺口只剩約 8.4 pp，小於等價界 ±10%，等價檢定在這一格測不到任何東西**（RL 與 estim 差 8% 也會被判「等價」），所以 E3 改成單邊優越性：H1：RL > estim-yawlim（snap=False）；主檢定只取 POS_STEP 三格（任務選擇見下方「主檢定任務的選擇規則」），Holm 校正，家族 α=0.05（3.2）。
- **主檢定任務的選擇規則（v0.4）**：E3 的主檢定任務 = 在探索 seed（offset 0）上，古典 estim-yawlim（snap=False）與 informed-offset-yawlim 之間**平均相對缺口最大**的任務（候選：T2、POS_STEP），在 line6–8 上平均。**套用結果：POS_STEP。** 依據（`PROGRESS.md`「E3 古典基準分任務重算」，探索 seed，每格 100 ep，estim / informed）：

  | 構型 | T2 | POS_STEP |
  |---|---|---|
  | line6 | 0.197 / 0.204 | 0.552 / 0.595 |
  | line7 | 0.158 / 0.161 | 0.509 / 0.565 |
  | line8 | 0.131 / 0.136 | 0.410 / 0.505 |

  T2 的絕對缺口 ≤ 0.007（相對約 2–4%），POS_STEP 為 0.043–0.095（相對約 7–19%，line8 最大）；POS_STEP 的平均相對缺口遠大於 T2。confirmatory seed 上的古典結果模式相同（POS_STEP 缺口 0.047/0.061/0.109，T2 ≤ 0.005），但決定只依探索 seed。
- **揭露（v0.4 改任務）**：v0.3 的主任務原為 T2，是根據 T2+POS 合併值選的；分任務重算後發現 T2 幾乎沒有缺口，所以 v0.4 改用上述規則。這次改動在任何 RL 策略於 E3 上評估之前完成，只依據古典結果，而且探索 seed 已足以決定，confirmatory seed 上的古典結果模式也相同。凍結（git tag）之後的任何改動，一律列為「偏離預先登記」，並在報告中逐條列出。
- **基準數字**：estim(snap=False)-yawlim 與 informed-offset-yawlim 在 POS_STEP 單獨（主）與 T2 單獨的 confirmatory seed 分數（estim / informed）已算出（`PROGRESS.md` 第 525–533 行「E3 古典基準分任務重算」，第 529／531／533 行，rl_eval 新碼，每格 100 ep，seed 900000）：POS_STEP line6 0.548 / 0.595、line7 0.498 / 0.559、line8 0.385 / 0.494；T2 line6 0.210 / 0.209、line7 0.182 / 0.181、line8 0.163 / 0.168。T2+POS 合併分數：凍結後於 confirmatory seed（offset 900000）執行，結果寫入 `results/p2/classical_confirm.md`；不改任何規則或比較對象（見 §14）。**判讀規則**：古典方法的結論以 confirmatory 為準；若與探索 seed 結論方向不同，兩者都報、以 confirmatory 為主。探索 seed、T2+POS 合併的舊值：line6 0.379 / 0.402；line8 0.269 / 0.321；grid10 0.383 / 0.402（`p3d_resolve.md`），僅作背景，不是凍結值。T2 單獨與 POS_STEP 單獨的分任務數字見下方選擇規則的表（`PROGRESS.md`「E3 古典基準分任務重算」）；上述 confirmatory 數字之外的部分由凍結後的腳本產生（見 §14）。
- **補回比例**（必報，附 CI）：R = (RL − estim) / (informed-offset − estim)，分母用 **POS_STEP** 的 (informed − estim)。分母是 confirmatory seed 上的古典常數，所以每個 seed 的 R_s 是 d 的線性函數；區間 = 對 R_s 的 seed 層 t 區間（主）與 BCa（敏感度）。僅當 estim 對 informed-offset 的缺口 ≥ 5 pp 才算，否則報「無法計算」並寫原因（沿用 v6.1 HP2 規定，`PLAN.md` v6.1 歷史段；指揮官決定 4：只當描述性指標，不做二分判定）。>100% 不截斷。
- **三種結果的解讀（事先固定）**：
  1. **RL 顯著優於**（該格經 Holm 校正後拒絕 H0）：寫成「在 E3 這一格（條件於固定評估集，Holm 校正後），RL 的分數高於相對於狀態回授的古典全開法（snap=False、estim+yawlim）；補回比例 R [CI]」。不寫成「RL 一般優於古典法」；若 RL 超過 informed-offset（R>100%），按第 10 節註明 informed 不是理論上限。
  2. **無差異**（Holm 校正後未拒絕，且不符合顯著劣於）：寫成「未能顯示 RL 優於古典；本檢定**不是**等價檢定，不得寫成 RL 追平古典」。補回比例與其 CI 照報，供讀者判斷效果量。
  3. **顯著劣於**（作為描述性事後檢查：90% 雙側區間上界 U < 0）：寫成「RL 在這一格低於古典全開法」；H1 本來是單邊，此項不是預先登記的假設檢定，只是描述。
- **判定**：3.3 乙 + 補回比例；RL 訓練分佈須含偏離中心（`--payload-offset-max 1`）。E3 的 offset 是訓練 DR 範圍的邊界點（見第 8 節的分離說明），只能稱為「訓練範圍邊界」，不是分佈外泛化。
- **POS_STEP 為唯一主檢定**（3 格 Holm）；T2 單獨與 T2+POS（串接）一律為次要，只報差與 CI，不判定、不入家族。結論措辭一律加「條件於固定評估集」，兩層 bootstrap 與 t 區間並列，判定以 t 區間為準。

### 5.c RQ-D 壓力測試：飛行中 attach / drop

- **事件**：中途掛上（0 → 15%）或脫落（15% → 0）；事件時刻 t ~ U(1.0, 3.5) s；掛點 random；中心 / 偏離中心兩種（offset 同 E3）。任務 T2、POS_STEP、HOVER。env seed：confirmatory 的 900002 / 900001 / 900003（T2 / POS_STEP / HOVER）。實作 `rl_eval.py --event`，逐 episode 與古典端（在 confirmatory seed 上重算）成對。
- **古典端基準**：B1-QP-estim-reset-yawlim 對知道事件的 informed-event-yawlim 的分數比 = 0.98–1.00（12 個任務 × 條件列，探索 seed）；偏離中心六列成對分數差 −0.0016 到 −0.0137，CI 不含 0 但量小。無重置的 estim-yawlim：中心 0.92–0.98、偏離中心 0.66–0.89；不知道 payload 的 B1-QP-yawlim：中心 0.50–0.70、偏離中心 0.07–0.09。出處：`results/p2/rqd_classical.md` 事實陳述 1–3。以上為探索 seed 數字；凍結值：凍結後於 confirmatory seed（offset 900000）執行，結果寫入 `results/p2/classical_confirm.md`；不改任何規則或比較對象（見 §14）。判讀規則同 5.a（以 confirmatory 為準，方向不同則兩者都報）。
- **指標**：整段分數（**描述性**：每格報估計值與 90% t 區間，並報 48 格中區間落在 ±B% 內的比例；基準 = estim-reset-yawlim；不逐格宣稱）；事件後 1 s 峰值位置誤差、恢復時間（未恢復以剩餘時間計，右設限）、成功率另報差值與 CI，不做二分判定。
- **已知限制（須在報告引用）**：T2 每 1.5 s 換目標，峰值與恢復時間混入換目標暫態，T2 的峰值／恢復時間不單獨解讀（`rqd_classical.md` 事實陳述 6）。env 事件仍是質點，與估計器同族。
- **古典公平性**：重置版的偵測超參數在驗證 seed 100+ 上 ≤20 組搜尋（`rqd_tune.json`）；評估用 confirmatory seed，與驗證 seed 不重疊。對等性見第 10 節。
- 兩種古典對手（estim 無重置、estim-reset）都當基準列出；描述性比較以 estim-reset 為準（最強古典法），無重置版當對照。

### 5.d 合併規則

- 5.a、5.b、5.c 各自報告；E1 各格不合併，E3 三格以 Holm 合為一個家族，不合併成單一總結論，也不在格之間挑最好看的。
- **主結論只依兩樣東西寫**：(i) E3 的 POS_STEP 三格 Holm 校正後的結果；(ii) E1 三格各自的等價結果。加上第 7 節的 go/no-go 與第 6 節的退化比作背景，並限定為「相對於狀態回授的古典法」、「條件於固定評估集」。E3 的 T2 單獨與 T2+POS、RQ-D 48 格、E2 都只作描述，不入主結論的判定句。

---

## 6. 必報的配套指標

### 6.1 標稱基線與退化比

- RL 同時報 payload=0 同格（同構型、同任務、同 seed）的標稱分數，以及**退化比 = RL(payload 下分數) / RL(標稱分數)**，與主古典對手的退化比並列（estim(payload)/estim(標稱) 由 `g2_estim.md` 與 `p3d_factorial.md` nominal 節取，confirmatory seed 重算）。
- 目的：把「RL 本來就弱」與「RL 不會處理 payload」分開。
- RL 標稱基線策略：`results/p3/line6_nom_k2_h4` 配方（無 payload 訓練）。排程見 `results/p3/STATUS.md`（#76/#82）。退化比同時用「payload DR 策略的標稱表現」與「無 payload 專訓策略」兩種分母，各報一次 `[推論/提案]`。
- 古典標稱基準：line8 T2 nominal B1-QP 0.2744、B1-QP-yawlim 0.3778（`p3d_factorial.md` 事實陳述 1；探索 seed；confirmatory 值：凍結後於 confirmatory seed（offset 900000）執行，結果寫入 `results/p2/classical_confirm.md`；不改任何規則或比較對象（見 §14））。

### 6.2 訓練/評估分佈分離

- 訓練 DR：質量比例 U(0, 0.20)，掛點在所有模組均勻隨機；偏離中心訓練（`--payload-offset-max 1`）；事件訓練（`--payload-event-prob 0.3`，attach/drop 各半，t_event ~ U(1.0, 3.5) s）。出處：`PLAN.md` v7.1 RQ-R；`PROGRESS.md` RQ-D 準備段。
- 訓練環境 seed：見 3.1（P4 訓練 seed 101–112，環境 seed = s×1000+i）；評估環境 seed 為 3.1 的 confirmatory 組，與訓練、驗證、探索分開。
- **E1**（固定點 15% argmax-x）、**E3**、事件格為主判斷格；**E2 留出集**（質量 {0.10, 0.15, 0.20} × 掛點每 episode 隨機，每個質量 100 episode）為**次要格，只報告、不判定**。E1 與 E2 結論不一致時兩個都報，不挑。
- E2 的 informed 參考不支援（需逐 episode host），E2 對手為 B1、B1-QP、B3 系列，以及 estim；estim 在 E2 與 informed 差 ≤0.004（`g2_estim.md` 留出集段）。`[凍結後於古典 confirmatory 重算時處理，§14(a)]` `rl_eval` 的 E2 古典端目前只重算 B1、B1-QP、B3 系列（`rl_eval.py:170-190`），E2 要補 estim-yawlim(snap=False) 才能報告差值；補不上就在 E2 報「無法計算」，不影響主判斷格。
- 古典法調參用驗證 seed（100+任務 seed，`g2_estim.md`），P4 判定用 confirmatory seed，兩者不重疊。

### 6.3 A3：古典法也吃帶雜訊觀測（必報，不在附錄）

目的：量化 2.1 的資訊不對稱對古典法的有利程度。
- 設定：在 confirmatory seed 上，把古典外迴路讀到的角速度 `state["w"]` 換成 `w + gyro_noise × N(0,1)`（與 RL 的 obs 相同分佈；其餘 pos/vel/q 與 RL 一樣是真實值），主古典對手 estim-yawlim（snap=False）與 informed 參考各算一次。`[不適用：P4 未開，A3 只服務 RL 比較]`：目前 `controllers.py` 沒有這個選項（只有 estim 的估計器內部用雜訊 w，`controllers.py:810`），須新增並測試。
- 報告：5.a、5.b 各格的 A3 版古典分數，以及 RL 對 A3 版古典的 d_s 與區間（同 3.2 方法）。**A3 不改變 5 節的判定**（判定仍以主古典對手為準），但主結論段必須並列 A3 的結果；若 A3 使結論翻轉，報告須明寫。

---

## 7. RL 配方 go/no-go（P4 開跑門檻）

**go/no-go 與 7.2 診斷一律使用驗證 seed（`--seed-offset 100`，3.1：100/101/102/103）評估，不得用探索 seed 或 confirmatory seed。**

### 7.1 Go/no-go 判準

- **G-main（機械判定）**：用**訓練 seed 101 的單一策略（`final.pt`）**，在**驗證 seed（`--seed-offset 100`，T2 的環境 seed = 102）**上跑 E1 line6 T2，100 個 episode，取平均分數；分數 ≥ 門檻即過，否則不過。**門檻 = estim-yawlim（snap=False）在同一驗證 seed、同一 100 個 episode 上的分數 × 0.9**。出處：`PLAN.md` v7.1「下一步」第 2 項。
- 門檻數字：上述 estim-yawlim 分數 × 0.9，數字：不適用：P4 未開（7.3），見 §14。（G-main 從未執行：7.3(b) 的開發 run 已判定不過，見 §14。）舊文件的 0.276（= 0.307 × 0.9）是 snap=True、探索 seed 的數字，**不是凍結值**。另算對無 yawlim 版（PLAN 字面「estim 0.210，即 ≥0.189」，0.210 是無 yawlim 版在 line6 E1 T2，`g2_estim.md` 第 15 行）的 90% 一併報告；P4 開跑的 go 條件取較嚴者（指揮官決定 1）。理由：RL 若沒有 yawlim，yaw 限速屬於 RL 要自己學的部分。
- 驗證 seed 與古典調參重疊（estim 超參數在 100+ 上調過，對古典有利），因此 go 門檻對 RL 是偏嚴，不是偏鬆。
- line6 E1 T2 informed 對 B3+adapt 只差 3.2 pp（探索 seed，`g2_estim.md` 第 15 行；PLAN v7.1 亦寫），line6 只當管線先導，主判斷靠 line7/8；go/no-go 以 line6 為準，因為 line6 的 RL 先導先跑。

### 7.2 失敗時的診斷順序（事先固定，次數有上限）

1. 先試較長的歷史 H=16（1 次訓練，**訓練 seed 201**）。
2. 再試 critic 加明確質量欄位（PRIV_DIM 4→8；`policy.py:31`；1 次訓練，**訓練 seed 202**，沿用上一步結果較好的 H）。
- 診斷用的訓練 seed 201、202 **不屬於 101–112**（正式 seed 不被診斷汙染）；診斷策略以 G-main 同一規則評估（驗證 seed、100 episode、同一門檻）。
- **診斷預算上限：2 次訓練（各 1 個 seed）**。PLAN v7.1 原寫 H「16、32」，本版收斂為 H=16 一次（理由：把診斷次數鎖在 2 次；H=32 不再試，若 H=16 無改善也不試）。`[推論/提案]` 此縮減請指揮官確認。
- 兩次診斷後 G-main 仍不過 → 停止 P4 並問使用者，不自行換方向。出處：`PLAN.md` v7.1「下一步」第 2 項；`PROGRESS.md` PLAN v7 白方 (6)。
- 每項診斷只做一次，結果（無論成敗）寫進進度檔。

### 7.3 P3a 管線檢查：目前未通過，P4 的開跑規則

- **事實**：P3a = TRAIN 四構型（single/line2/line3/square4，各 T1+T2）RL ≥ B1 的 90%（每格）。`train4_mix_k1_h4`（3e8 步）結果：12 格中 6 格 < 0.90（勘誤：應為 4 格，見 §11 v1.0-勘誤 1），T2 比值 single 0.937、line2 0.816、line3 0.693、square4 0.806，另 line3 ATT_STEP 0.871。出處：`results/p3/STATUS.md`「train4_mix_k1_h4 評估結果」；`results/p3/train4_mix_k1_h4/eval_final.json`。（這些是探索 seed 上的數字。）
- 訓練中 T2 成功率 7.5e7 步到 3e8 步停在 0.70–0.86，是平台不是步數不足（指揮官補充，同 STATUS）。單機 k=0 H=1 T1+T2 混合 T2 成功率 98% 作對照。
- **進行中**：k/H 消融 #81 `abl_single_mix_k1_h4`（single:T1+T2、k=1、H=4、1.5e8 步），之後 #82 line6_nom_k2_h4、#83 line6_pl20off_ev_k2_h4。判讀規則（STATUS 已寫）：#81 T2 成功率 ≈98% → k/H 無害，問題在多構型共用；#81 明顯低 → 問題在 k/H，先修配方再跑 #82/#83。消融結果：#81 顯示 k/H 無害，train4 的平台來自多構型共用（`PROGRESS.md` 2026-10-31「RL 配方：line6 學不起來」段）；其後 line6 在兩次配方修改（mean-bound、team-sample）內未過，P4 依本節規則不開（§14）。
- **P3a 未過時 P4 能否開跑的規則（指揮官決定 2，採納）**：
  1. P3a 的目的只是「訓練管線沒有 bug」。P4 的結論只建立在 P4 自己的判斷格（line6–8）上，所以 P3a 在 TRAIN 四構型上的 90% **不是 P4 的必要條件**，但必須有替代的管線證據。
  2. 替代條件（全部滿足才可開 P4）：
     (a) **單機配方**＝ single:T1+T2 混訓的配方（#81 `abl_single_mix_k1_h4` 或其後續；「後續配方」定義為**最終凍結的那一個配方**）；在 rl_eval 的 ATT_STEP、POS_STEP、T2 三個任務上，以**驗證 seed（offset 100）**計，**每格分數 ≥ 該格 B1 分數的 0.90**（B1 為同驗證 seed、同 episode 重算）；
     (b) 7.1 的 G-main 通過（line6 E1 T2 ≥ 90%）；
     (c) 沒有通過 P3a 這一事實須寫進報告「已知限制」，並在報告中列出 12 格的實際比值。
  3. 若 (a) 或 (b) 任一不過，**不開 P4**，依 7.2 診斷；7.2 也不過就停下問使用者。
  4. 不允許「重新定義 P3a 門檻」使其變成通過；若要改門檻，須在凍結前寫入修訂紀錄。
  5. P3a 未過時 P4 的訓練配方（k、H、步數）須在看到 confirmatory 結果前定死，並寫入第 8 節。

### 7.4 執行順序表（凍結前，依序執行；每步完成才進下一步）

| 步 | 內容 | 填什麼 `[待填]`／產出 | 由誰填 |
|---|---|---|---|
| 1 | k/H 消融 #81（→ #82、#83 依 7.3 判讀規則） | 7.3 的消融結果 | 指揮官，步驟 1 完成時 |
| 2 | 定配方（k、H、步數、lr、ent_coef、n_epochs、PRIV_DIM） | 不適用：P4 未開（7.3），見 §14 | — |
| 3 | go/no-go：G-main（訓練 seed 101，驗證 seed）；不過才做 7.2 診斷（seed 201、202，至多 2 次）；並檢查 7.3(a) | 不適用：P4 未開（7.3），見 §14 | — |
| 4 | **配方凍結**（之後不得再改配方） | 不適用：P4 未開（7.3），見 §14 | — |
| 5 | 先導 SD（seed 101–103、凍結配方、驗證 seed）→ 決定 n 與 B（3.4 機械規則） | 不適用：P4 未開（7.3），見 §14 | — |
| 6 | 古典基準在 confirmatory seed（offset 900000）上重算（E1/E2/E3/event、E3 的 POS_STEP 與 T2 分任務基準、6.1 標稱基準） | 改為凍結後執行的確認性報告（§14(a)），不阻擋凍結；E3 分任務數字已填入 5.b | 凍結後腳本 |
| 7 | 與古典端相關的待實作項完成並通過測試（`rl_eval --seed-offset`、`tests/test_rl_eval_seed.py`，commit 5b4fe04）→ git tag `prereg-v1.0` 凍結 | 第 8 節 git 凍結點 | 指揮官，步驟 7 |
| 8 | P4（RL 策略在 confirmatory seed 上評估） | 不適用：P4 未開（7.3），見 §14 | — |

- G-main 的 seed 101 策略同時是步驟 5 先導的第一個點，不重複訓練。
- **先導 SD 只有約 3 個點，是粗估**（見 3.4）。
- 步驟 4 之後若要改配方，須在修訂紀錄揭露，且該次結果標為「探索性」。

---

## 8. 凍結清單（P4 前須填的值）

| 項目 | 值 | 狀態 |
|---|---|---|
| RL 配方（k, H, 步數, lr, ent_coef, n_epochs, PRIV_DIM） | 不適用：P4 未開（7.3），見 §14 | 不適用 |
| 訓練 DR（payload-max, offset-max, event-prob） | 0.20 / 1 / 0.3（`PROGRESS.md` RQ-D 準備段，#83 參數） | 暫定 |
| 訓練 seed | 主要 101–108；延長 109–112；診斷 201、202（3.1、7.2） | 已定 |
| 驗證 seed | +100：100/101/102/103（3.1） | 已定 |
| 探索 seed | +0：0/1/2/3（3.1） | 已定 |
| confirmatory seed | +900000：900000/900001/900002/900003，n_eps=100 | 已定 |
| n 與等價界 B | 依 3.4 規則由先導 SD 決定（n∈{8,12}，B∈{10%,15%}） | 不適用：P4 未開（7.3），見 §14 |
| 統計方法 | 主：seed 層 t 區間（條件於固定評估集）；並列：兩層 bootstrap；敏感度：BCa（n=8 不可靠）（3.2）。多重比較：E3 的 POS_STEP 三格 Holm α=0.05；E1 各格分開；RQ-D 描述性 | 已定；兩層 bootstrap：不適用（P4 未開） |
| checkpoint | `final.pt`，不挑選 | 已定 |
| estim 設定 | g2_estim_tune.json 第 8 組 + snap=False；reset 版 rqd_tune.json | 已定 |
| yawlim 參數 | rate_c=0.556、yaw_ratio=0.415 | 已定 |
| 古典端基準數字（E1/E2/E3/事件，snap=False，confirmatory seed） | 凍結後於 confirmatory seed（offset 900000）執行，結果寫入 `results/p2/classical_confirm.md`；E3 POS_STEP／T2 分任務已填入 5.b | 凍結後確認性報告（§14(a)） |
| go 門檻數字 | 不適用：P4 未開（7.3），見 §14 | 不適用 |
| 待實作清單 | `rl_eval --seed-offset`（含快取 key 與 meta 加 seed）與 `tests/test_rl_eval_seed.py`；古典端加 estim-yawlim 於 E1/E2/E3/event；E3 POS_STEP 與 T2 分任務基準；兩層 bootstrap + 測試；A3 控制器選項 + 測試；seed 斷言 | `rl_eval --seed-offset` 與 `tests/test_rl_eval_seed.py`：已完成（commit 5b4fe04）；兩層 bootstrap、A3 控制器選項（只為 P4 服務）：不適用（P4 未開）；古典端 estim-yawlim 與 E3 分任務基準：隨確認性報告（§14(a)） |
| git tag 凍結點 | `prereg-v1.0`（由指揮官在 commit 後打 tag） | 7.4 步驟 7 |

E3 offset 與訓練分佈的關係：訓練 offset 為 x、y ∈ U(−s·0.0333, s·0.0333)、z ∈ U(−s·0.0215, 0)（s=1），E3 的固定點 (+0.0333, −0.0333, −0.0215) 位於此範圍的角落，屬訓練支撐集的邊界而非留出。因此 E3 檢驗的是「訓練分佈邊界上的極端點」，**不是**分佈外泛化。若要檢驗留出偏移，須另設一個訓練 DR 範圍之外的點，並在凍結前寫入 `[推論/提案]`；目前不做。

---

## 9. 附錄（不列入主判斷，結果單獨報、不影響第 5 節結論）

- **A1 RQ-S 練小飛大**：TRAIN 構型 + payload DR 訓練，零樣本到 line6–8。
- **A2 TWR × yaw 權限掃描**（原 v5.1 主打，降為附錄）；缺口對構型長度圖 `results/p2/g1_coord_gap_vs_length.png`。
- **A4 MuJoCo 重現**：只驗積分器，不處理估計器與 env 共用 payload 模型的循環。
- **A5 RL residual-on-estim**：RL 只學 estim 輸出的殘差，對照純 RL 與純 estim。
- 其他：飛行影片、ATT_STEP 的 3–5 pp 局部缺口（已放棄去中心化）、RQ-C 選做。
- （A3 已於 v0.2 升為必報項目，見 6.3。v1.0：A3 只用於 RL 對古典的公平比較，P4 未開故不適用，見 §14。）

---

## 10. 異常處理與揭露義務

- 所有主判斷格（5.0 清單）的結果必須報告，包含負面；無法計算的格報「無法計算」與原因。
- 功效分析（3.4）決定的 n 與 B、任何修訂（下節）必須揭露。
- 若 RL 超過 informed：informed 不是理論上限（α 限幅沿用標稱；all15 的 informed 不知推力係數與對接偏差；`PLAN.md` 已知限制），報告寫明。
- **調參預算並列與對等性**（報告必列）：

| 方 | 預算 |
|---|---|
| 古典端 | estim 20 組、yawlim 20 組、重置 20 組（共 60 組；每組在驗證 seed 的 32 episode 上評估，`g2_estim.py:32` VAL_EPS=32），每次評估成本低 |
| RL 端（本次 PLAN 允許） | 重跑／消融：#81、#82、#83 共 3 次訓練（已排程），之後配方修改（k、H、步數）至多 2 次；go/no-go 失敗後的診斷至多 2 次（7.2，訓練 seed 201、202）；總計至多 **7 次訓練**，每次 1.5e8–3e8 步 |

  **不對等**：古典端 60 組是低成本、較多自由度的超參數搜尋；RL 端 ≤7 次是高成本、較少組數。名目次數 RL 較少（對 RL 不利），但每次調整的自由度（配方）也更大；兩者沒有可換算的單位，不宣稱對等，只並列並請讀者自行判斷。RL 在超出上述次數後不得再調，否則該次結果須標為「探索性」。
- 任何在凍結後才加入的 seed、構型、對手，標為「探索性」，不用於第 5 節判定。

## 11. 修訂紀錄

| 日期 | 版本 | 修訂 | 理由 |
|---|---|---|---|
| 2026-10-30 | v0.1 | 初稿 | |
| 2026-10-30 | v0.2 | 白方第 1 輪的修改：(1) seed 分驗證／探索／confirmatory 三組並寫死清單，訓練 seed 101–108＋延長 109–112，檢查環境 seed 公式不撞（3.1）；(2) 判定分格：E1 等價檢定、E3 改單邊優越性＋補回比例＋三種結果解讀、RQ-D 維持（3.3、5）；(3) n 在功效分析一次定死，延長只由先導 SD 觸發，所有 seed 都報、發散記失敗、一律 final.pt（3.2、3.4、3.5）；(4) 統計主方法改 seed 層 t 區間，BCa 與兩層 bootstrap 作敏感度，兩層 bootstrap 標為待實作（3.2）；(5) 0.307／0.276 標為待重算，RL 調參預算具體化並與古典 20/20/20 並列（4、5.a、7.1、10）；(6) E2 降為次要、主判斷格清單寫死不得刪格（5.0）；(7) 資訊不對稱升入正文 2.1，結論限定為「相對於狀態回授的古典法」，A3 升為必報（6.3）；(8) 揭露「snap=False 是看過偏離中心診斷後才選」（4）與「E3 是看過古典缺口分解後才選，在任何 RL 評估之前、只依據古典結果」（5.b） | 白方審查必改項，指揮官採納 |
| 2026-10-30 | v0.3 | 白方第 2 輪（使用者選「全改後凍結」、「T2 為主，3 格 Holm」）：(1) 多重比較：E3 主檢定只取 T2 三格、單邊優越性、Holm α=0.05；E3 的 T2+POS 降為次要；E1 三格各自等價、不合併；RQ-D 48 格改描述性；總結論只依 E3 校正後與 E1 等價結果（3.2、3.3、5.0、5.b、5.c、5.d）；(2) 快取與 seed：`--seed-offset`（驗證 100／confirmatory 900000／探索 0）、快取 key 與 meta 含 seed、offset≠0 不得沿用 `g2_raw.json`、`tests/test_rl_eval_seed.py` 為凍結條件（3.1）；(3) go/no-go 機械化：G-main 用訓練 seed 101 final.pt、驗證 seed、100 episode、門檻 = estim-yawlim 90%；7.2 診斷改用訓練 seed 201、202；7.3(a) 單機配方具體化為 ATT_STEP／POS_STEP／T2 每格 ≥ B1 的 0.90，「後續配方」＝最終凍結配方（7.1–7.3）；(4) 新增 7.4 執行順序表與 `[待填]` 的填寫者（指揮官），並註明先導 SD 約 3 點為粗估（3.4、7.4、8）；(5) 固定評估集：t 區間只涵蓋訓練 seed 變異，5.a／5.b 結論加「條件於固定評估集」，兩層 bootstrap 與 t 區間並列、判定仍以 t 區間，BCa 在 n=8 不可靠（3.2、5.a、5.b）；(6) 摘要揭露「E3 與 snap=False 皆看過古典結果後才選」（文件開頭） | 白方第 2 輪必改項與建議項，使用者決定全改後凍結 |
| 2026-10-30 | v0.4 | E3 主檢定任務由「T2」改為規則：探索 seed 上 estim-yawlim 與 informed-offset-yawlim 平均相對缺口最大的任務（候選 T2、POS_STEP），套用結果 POS_STEP；主檢定 = POS_STEP 的 line6/7/8 三格單邊優越性、Holm α=0.05；T2 單獨與 T2+POS 降為次要；補回比例分母改用 POS_STEP 的 (informed−estim)（摘要、3.2、3.3、5.0、5.b、5.d、7.4、8）。揭露：v0.3 選 T2 是看 T2+POS 合併值；分任務重算後 T2 缺口 ≤ 0.007；改動在任何 RL 於 E3 評估之前、只依古典結果，探索 seed 已足以決定、confirmatory seed 模式相同；凍結後任何改動列為「偏離預先登記」並於報告逐條列出 | `PROGRESS.md`「E3 古典基準分任務重算」；使用者同意改任務 |
| 2026-11-01 | v1.0 | 凍結（tag 首次誤打在未修完的 0a8b84f，數分鐘後、任何 confirmatory 結果產生前改打在本修訂；差異只有待實作標記與 A3 不適用的一致化）。(1) P4 主實驗依 7.3 不開（line6 在兩次配方修改內未過 go/no-go；正式 G-main 從未執行），只為 P4 服務的 `[待填]`（3.4 的 SD_rel/n/B、7.1 go 門檻、7.4 步驟 2–5、第 8 節對應列）改標「不適用：P4 未開（7.3），見 §14」，原文規則保留；(2) 古典端 confirmatory 重算（5.a、5.b、5.c、6.1、第 8 節）改為凍結後執行的確認性報告，E3 POS_STEP／T2 分任務 confirmatory 數字已填入 5.b；(3) 待實作清單標註完成／不適用；(4) git tag 定為 `prereg-v1.0`；(5) 新增 §14 | 使用者決定；規則與比較對象一律不變 |
| 2026-11-02 | v1.0-勘誤 1 | 7.3 事實段「train4 12 格中 6 格 < 0.90」改正為 **4 格**（line2／line3／square4 的 T2 與 line3 ATT_STEP；同段列出的 single T2 0.937 本來就 ≥ 0.90）。原文保留，只在 7.3 該句後加指向本列的記號，以本列為準。P3a 未過的判定不變。§14(b) 的 seed 8 結果補在 §14 | 重新計算 `results/p3/train4_mix_k1_h4/eval_final.json` 時發現；只是事實更正，不改任何規則 |

## 12. 指揮官決定（2026-10-30；v0.3 已與 3–11 節規則對齊）
1. go/no-go 門檻（7 節）：兩個版本都報；P4 開跑的 go 條件取較嚴的 **estim-yawlim 的 90%**（E1 line6 T2，驗證 seed 102，snap=False；v0.3：用訓練 seed 101 的 final.pt、100 episode 機械判定）。數字：不適用：P4 未開（7.3），見 §14；舊稿的 0.276 是 snap=True 探索 seed 數字，不當凍結值。理由：主對手就是 estim-yawlim，yaw 限速是 RL 必須自己學會的東西。
2. P3a 未過時的規則（7.3）：採納提案。P3a 不是必要條件；P4 要開跑，必須同時滿足：單機配方 ≥ 90%，且 line6 E1 T2 過上一條的 go 門檻。兩者有一項沒過，P4 就不開。（v0.3：「單機配方 ≥ 90%」＝7.3(a) 的 ATT_STEP／POS_STEP／T2 每格 ≥ B1 的 0.90。）
3. 功效分析放寬界值：上限 ±15%，比照 Hopper，只能放寬一次（3.4 的機械規則；與 n 一併一次定死，只依先導 SD）。
4. 補回比例的 5 pp 門檻：只當附帶的描述性指標，不做二分判定（E3 必報，附 CI，缺口 <5 pp 報「無法計算」）。v7.1 取消的是 HP2 的判定地位，這個描述性用法保留。
5. estim 主設定：定為 **snap=False + yawlim**。理由：p3d_factorial 和 rqd 顯示 snap=True 在偏離中心時失效，snap=False 則很穩健。這個設定是在看過偏離中心診斷後才選的，報告必須揭露（已寫入第 4 節與修訂紀錄）；snap=True 放進「什麼沒成功」。
6. 待補的程式：rl_eval 的古典端加入 estim-yawlim（E1/E2/E3/event 都要），重算 E3 的 POS_STEP 與 T2 分任務基準（v0.4），並加 `--seed-offset`、兩層 bootstrap、A3 控制器選項（含測試）。凍結前完成。
7. E3 是訓練 DR 範圍的角落點，不是留出點：只能稱為「訓練範圍邊界」，不能宣稱是分佈外泛化。
8. （v0.2 新增，派工單）seed 三組分離與清單、判定分格（E1 等價、E3 單邊優越、RQ-D 維持）、n 一次定死且延長只由先導 SD 觸發、統計主方法 seed 層 t 區間、E2 為次要格、主判斷格清單不得刪格、A3 必報、E3 選格過程需揭露。與舊稿衝突時以此條為準。
9. （v0.3 新增，白方第 2 輪，使用者決定「全改後凍結」）多重比較：E3 只取 T2 三格、單邊優越性、Holm α=0.05；E3 T2+POS 次要；E1 各格分開；RQ-D 48 格描述性；總結論只依 E3 校正後與 E1 等價結果。G-main 與診斷、7.3(a) 的機械條件見 7.1–7.3；執行順序見 7.4，所有 `[待填]` 由指揮官依 7.4 填。與前面條目衝突時以此條為準。
10. （v0.4 新增，使用者同意）E3 主檢定任務改用規則選定（探索 seed 上平均相對缺口最大者），結果為 POS_STEP；T2 與 T2+POS 為次要。與第 9 條及前面條目中「E3 只取 T2」的字樣衝突時，以此條為準。

## 13. 探索性延伸：殘差 RL（2026-10-31，凍結前加入，不屬判定）
- 背景：依 7.3，RL 主實驗不開（line6 標稱在 2 次配方修改後仍未達標，診斷見 results/p3/STATUS.md）。
- 使用者決定另外做殘差 RL：u = clip(u_base + scale·δ)，u_base 是 B1-QP-estim-yawlim（snap=False），RL 輸出 δ。
- 限制：只用驗證 seed（offset 100）和探索 seed（offset 0），**不使用 confirmatory seed**，不進第 5 節判定，報告一律標「探索性」。主要觀察格是 E3 POS_STEP line6–8（古典缺口所在）。

## 14. 凍結說明（v1.0，2026-11-01）
- **P4 不開的依據**：依 7.3(b)，P4 開跑需 G-main（7.1）通過。line6 在兩次配方修改（#82 原版之後的 mean-bound、team-sample）內未過 go/no-go；這些都是訓練 seed 7 的開發 run（出處 `results/p3/STATUS.md`、`PROGRESS.md` 2026-10-31 各段），預算用盡，故 RL 主實驗不開，主結論照負面寫。**誠實說明**：正式的 G-main（訓練 seed 101、驗證 seed、100 episode）從未執行，因為 7.3(b) 的開發 run 已判定不過；因此 3.4 先導 SD、n、B、go 門檻數字、RL 配方凍結值皆未決定，標「不適用：P4 未開」，原文規則保留。沒有任何 RL 策略在 confirmatory seed 上評估過。§13 的探索性殘差 RL 已跑，結果見 `results/p3/STATUS.md`（驗證 seed：E3 POS_STEP 0.554 vs estim-yawlim 0.565，未補缺口），標探索性、不作判定。
- **凍結後才執行的項目**（皆不改任何規則或比較對象）：
  - (a) **古典 confirmatory 重算**：E1／E2／event、6.1 標稱基準與 5.b 的 T2+POS 合併值，於 confirmatory seed（offset 900000）執行，結果寫入 `results/p2/classical_confirm.md`。E3 的 POS_STEP／T2 分任務 confirmatory 數字已於凍結前算出並填入 5.b。**判讀規則**：古典方法的結論以 confirmatory 為準；若與探索 seed 結論方向不同，兩者都報、以 confirmatory 為主。**結果（2026-11-02）**：384 格完成，方向全部與探索 seed 相同；機械規則標「不一致」的 9 列，confirmatory 差都 ≤ 0.006：4 列 CI 含 0、2 列下界為 0、3 列（RQ-D estim-reset 對 informed-event）為 +0.002 到 +0.003、CI 不含 0，方向是 informed 略好，不改變任何結論（2026-11-02 更正：原寫「CI 都含 0」有誤）。E3 十二格與 5.b 已填值相差 < 0.002。T2+POS 合併 estim／informed：line6 0.379／0.402、line7 0.340／0.370、line8 0.274／0.331。6.1 標稱 line8 T2：B1-QP 0.328、B1-QP-yawlim 0.411。RQ-D 四構型平均：estim-reset 為 informed-event 的 0.98–1.00 倍；無重置 estim 中心 0.89–0.98、偏離中心 0.67–0.89；B1-QP-yawlim 中心 0.45–0.70、偏離中心 0.06–0.10。T2 絕對分數比探索 seed 高約 0.02–0.05。依判讀規則，古典結論以這些數字為準，方向不變。
  - (b) **探索性 RL 補 seed**：訓練 seed 8，重跑殘差 RL（`res_line6_off_k2_h4` 同參數，gpujob #254）與 mean-bound 配方（`line6_nom_mb_k2_h4` 同參數，#255），於驗證 seed 評估；標探索性、不作判定。凍結時結果尚未產生。seed 8 不屬於正式 seed 101–112，與開發 run 的 seed 7 同系列。**結果（2026-11-02，探索性）**：殘差 RL seed 8 與 seed 7 各格分數差 ≤ 0.002（兩者都在 120 分鐘上限停在約 3.3e7 步；E3 POS_STEP 0.555，對 estim-yawlim 0.565，成對差 −0.010 [−0.011, −0.009]）；mean-bound 配方 seed 8 為 ATT／POS／T2 = 0.634／0.707／0.661 × B1（seed 7 是 0.491／0.892／0.929），仍未達標。出處：`results/p3/STATUS.md`「seed 8 補跑」。
