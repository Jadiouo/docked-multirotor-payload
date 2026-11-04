# 相關工作筆記（2026-10-24，來自 subagent 調查）

「全文」表示有打開 arXiv HTML 讀過；「摘要」表示只讀了摘要或搜尋結果的摘要。結論類的句子都是推論，投稿前要用 Google Scholar 再補查一輪。

## 模組化飛行器（全部是古典控制，沒有 RL）
- ModQuad（Saldaña 等，ICRA 2018）摘要：https://www.lehigh.edu/~das819/pdf/icra18-modquad.pdf
- H-ModQuad，arXiv 2202.00788（2022 年發表，2025/08 更新 v3）全文：每個模組用 pseudoinverse 區塊獨立算出自己的推力，測了 1、2、4 個模組。假設對接已經完成，不討論分離。這是本專案的古典 baseline。
- MARS-Dragonfly，arXiv 2604.05499（2026）讀了一半全文：MPC 加 allocator，在 Gazebo 上模擬 10 多種構型，有磁吸硬體。
- TransforMARS，arXiv 2509.14025，全文：自重組規劃，沒有學習。

## 單一 policy 控制不同的 multirotor
- **NTNU，arXiv 2606.10857（2026/06）全文：這是最接近的前人工作。**
  - 方法：PPO 加 GRU，以 B_norm allocation matrix 當條件。
  - 範圍：4 旋翼和 6 旋翼各自一個家族，馬達數固定。
  - 成功率：通用 policy 85%／80%，專屬 policy 98%／97%。
  - 他們把「變動轉子數」列為未來工作，也認為 graph 或 transformer 類的方法不適用。
- arXiv 2503.09622（IROS 2025）全文：單剛體輸出 CTBT，構型編成 10 維參數，靠 RMA 式的適應模組處理。泛化的是尺度，不是轉子佈局。
- RAPTOR，arXiv 2509.11481，摘要：2084 個參數的 recurrent policy，10 台實機，都是一般四旋翼。
- Rehberg 等，arXiv 2602.15533（RA-L 2026）全文：policy 輸出 wrench，再交給學習式 allocation network。

## 形態通用 RL（方法源頭）
- Huang 2020，arXiv 2007.04976，摘要：每個模組共享同一份 policy，沿樹狀結構傳遞訊息。
- Shared Modular Recurrence，arXiv 2506.08630，全文；GCNT，arXiv 2505.15211，摘要。都只做 locomotion，沒有飛行器。
- 模組化月球機器人去中心化 RL，arXiv 2510.20347，全文：沒有拿留出的構型做量化評估，也沒有 scaling 模組數。

## 對接與分離
- Tao 等 2024，摘要：兩台機的 offline-to-online RL 對接。
- 找不到任何用 RL 處理「對接後剛性結構在對接或分離瞬間」控制的論文。

## 早期調查（多機吊掛搬運，已讓給 03）
- 2508.01522、CrazyMARL 2509.14126、MuJoCo-Drones-Gym 2606.08039、2412.11764、2311.13081、OmniDrones 2309.12825、Crazyflow 2606.01478

## 補查（2026-10-24 第二輪，全部只讀了摘要）
- 找不到任何論文做過「資訊量梯度」（L0→L3 式）的比較。NTNU 和 RAPTOR 都只比較「有構型描述」和「沒有構型描述」兩種情況。
- **UFO（Mu & Chirarattananon，arXiv 1911.03615，T-RO 2020）**：用多個 IMU 快速估計模組化多旋翼的構型，再用自適應幾何控制器在線上精修參數。這是最強的古典對手，也是 B4 的首選。它「不需要知道模組位置」這一點還沒有核對，要讀全文確認。
- 推薦的 INDI（B3）文獻：
  - 控制律：Smeur 2016（JGCD，adaptive INDI，線上估計 G）、Smeur 2017（arXiv 1701.07254，cascaded INDI）
  - 實作參考：Sun 2020（arXiv 2002.07837）
  - INDI 可以按 G 的來源分級：G 由已知的 allocation 算出，對應 L3；G 用 RLS 線上估計，對應 L0。這樣古典方法也能做一條資訊梯度。
- 單機用 RLS 估計 control effectiveness：Blaha 2024（arXiv 2406.11723、2409.01080）。
- 類比工作：Whitman 2021（arXiv 2105.10049），依據機器人的設計圖共享模組權重，在沒見過的設計上做零樣本泛化。

## UFO 與 NTNU 全文核對（2026-10-24，已讀全文，引句已抽查）
- **UFO（1911.03615）**
  - 不需要事先知道模組的位置，但仍需要以下先驗：
    - 模組數 N
    - 各模組的質量和慣量（取自 CAD）
    - 推力方向互相平行
    - 酬載是平板
    - 質心大致在幾何中心
  - 構型的估計方式：
    - 用全部 N+1 個 IMU 集中式估計
    - 要**手持全軸轉動 90 秒以上**，不是在飛行中估計
    - 原文："only the IMU from the control module is used for flight control"
  - 估出的值只當初值，要靠 adaptive geometric controller 線上修正。把 adaptive gain 關掉時，"crashes within 30 s"。
  - 和本專案的差別：UFO 是集中式，需要一段辨識階段，而且只處理平行推力和平板酬載。本專案問的是「沒有辨識階段、純分散式時，k 要多少」，UFO 可以當作集中式的對照。
- **NTNU（2606.10857）**
  - Limitations 寫到："currently designed for fixed-motor-count systems"
  - 附錄 F 有 descriptor 擾動的敏感度分析。
  - 沒有找到 payload 或質心偏移的實驗。
  - Related Work 認為 graph 或 transformer "not a natural representation for multirotor systems where control relies on the force-torque coupling of a single rigid body"。
    - 本專案的對接構型也是單剛體，所以論文中必須說明：GNN 對應的是模組之間的接合拓撲（以及感測和資訊的局部性），不是關節運動學。

## 接頭內力（2026-10-25，見 docs/joint_design_notes.md）
- LEGION（arXiv 2605.19431，全文）：把相鄰模組的 contact wrench 當成線性系統來估計和控制，超過 allowable wrench threshold 就判定為對接成功或失敗。屬於大型機，使用古典控制。
  - 原文："previous work ... still ignores the interaction force and torque—i.e., the contact wrench—between interlocked modules during flight"
  - 本專案的區別：Crazyflie 級的磁吸接頭、學習式 policy，以及把內力當成「不知道形狀的代價」來分析。
