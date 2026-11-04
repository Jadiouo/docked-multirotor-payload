# 模組化多旋翼環境規格（P2，v0，2026-10-25）

## 設計決定：動力學用批次剛體積分，不直接用 MuJoCo 當 RL 的模擬器
- 對接後整架是**單一剛體**，沒有接觸也沒有關節。Newton-Euler 方程可以直接寫出來，用 torch 對上千個 env 一起積分。這在 CPU 上比逐一 step MuJoCo 快一到兩個數量級（推論；實作後要量 sps）。
- MuJoCo 有兩個用途：
  1. **參考解**：給相同的推力序列，比對兩邊軌跡，用來驗證我們的積分器。
  2. **渲染**：用 CAD 的 STL 做外觀，產生 GIF。
- 地面：不建模接觸。高度 < 0 就判定墜毀，episode 結束。
- 理由寫進 PLAN 的 P2 段。

## 參數
- 模組：`cad/out/module_props.json`，包含質量、慣量、間距 pitch 和 joint_limits。
- 馬達：沿用 `quadsim/params.py` 的 CF2X 參數：a、Tmax、km_ratio、tau（一階延遲 0.07 s）。
- 組裝：用 `quadsim/assembly.assemble(cells, p, pitch)` 算出 M、J、rotor_pos、A。payload 和對接偏差用額外參數加進去，見「物理誤差」一節。
- 時間：物理 dt = 0.002 s；控制 100 Hz，也就是每個控制步做 5 個物理子步。
- 積分：位置和速度用半隱式 Euler；姿態用四元數，乘上 exp(ω dt) 後正規化。

## 狀態（世界座標，z 朝上）
p、v、四元數 q（body→world）、ω（機體座標）、馬達實際推力 T_act（4N 維），以及馬達指令。

每個子步：
1. 馬達一階延遲：T_act += (T_cmd − T_act)·dt/τ，並 clip 到 [0, Tmax·k_i]。k_i 是推力係數誤差，預設 1。
2. 合力與合力矩：
   - 機體座標的力 F = Σ T ẑ
   - τ = Σ r_j × (T_j ẑ) + Σ km·spin_j·T_j ẑ
3. 平移：m·dv = R F dt − m g ẑ dt
4. 轉動：J dω = (τ − ω × Jω) dt
- r_j 是真實的轉子位置（相對真實 CoM），含對接偏差和 payload 造成的 CoM 移動。

## 觀測：每個模組各一份，形狀 [B, N, obs_dim]
所有模組共用整架的導航狀態。對接後就是一個主體，由 leader 廣播，見 PLAN v4。

| 項目 | 維度 | 說明 |
|---|---|---|
| 位置誤差 R^T(p_goal − p) | 3 | 機體座標 |
| 速度 R^T v | 3 | |
| 姿態：R 的前兩欄 | 6 | 連續表示 |
| 角速度 ω | 3 | 加陀螺儀雜訊 |
| 本模組加速度計 | 3 | 比力，見下方公式 |
| 目標 yaw 誤差 sin、cos | 2 | |
| 四方向的鄰居旗標 | 4 | +x、−x、+y、−y 有沒有鄰居 |
| 上一步自己的 4 個動作 | 4 | |

- 加速度計的比力：R^T(a_c + g ẑ) + α × r_i + ω × (ω × r_i)。r_i 是本模組中心相對 CoM 的位置，**policy 拿不到 r_i**，加速度計讀數只是間接含有這個資訊。
- 盲飛條件：鄰居旗標全部設為 0。
- k 跳的資訊由 policy 的 GNN 透過鄰接矩陣取得，環境另外提供 `adjacency`，形狀 [N, N, 4]，帶方向。
- 雜訊：陀螺儀 σ = 0.01 rad/s，加速度計 σ = 0.05 m/s²。數值是假設，寫成參數。

## 動作：形狀 [B, N, 4]，範圍 [−1, 1]
- 馬達指令 T_cmd = Tmax·(u + 1)/2。
- policy 輸出經 env 端 clip，和 ppo.py 的預設相同。

## 任務（共用 reward 介面）
- **T1 恢復**：初始姿態隨機（roll、pitch 在 ±60° 內），初速和角速度也隨機，目標是回到原點並 hover。
- **T2 設定點跟蹤**：每 1.5 s 換一次目標，位置在 ±1 m 內、yaw 在 ±90° 內，要求快速到達。主實驗用 T2。
- **評估用的激烈機動**（G1 和主表）：
  - 大角度姿態階躍：roll 或 pitch 0.8–1.2 rad 的目標姿態
  - 1.5 m 的位置階躍
- episode 長度：5 s，也就是 500 步。時間到算 truncated。
- 終止條件：高度 < 0、|位置誤差| > 3 m、傾角 > 100°，任一成立即終止（terminated）。
- reward（每步，整隊共用一份）：
  - 主項：−(w_p|e_p| + w_v|v| + w_ω|ω| + w_yaw|e_yaw|)
  - 動作平滑：−w_u|u − u_prev|²
  - 存活獎勵：+c
  - 終止：−C
  - 選用：接頭負載懲罰 −w_j·max(負載比)
  - 權重寫在 config，初始值由實作者訂，要能讓 B1 在 T2 拿到合理分數。

## 物理誤差（只在評估時開啟，也可以在訓練時開啟做 DR）
- payload：質點，質量比例和掛點都是參數。
- 對接偏差：每個模組的真實位置加上 ±δ 的偏移，加上繞 x、y 的小傾斜 ±θ（推力方向跟著傾斜）。
- 推力係數：每顆馬達 k_j ~ U(1 − ε, 1 + ε)。
- policy 和古典法拿到的都是**標稱**形狀（鄰居圖），真實幾何只用在動力學裡。

## 接頭內力
- 每個構型預先算好一個線性映射 L：每個模組的 d'Alembert 外載（6N 維）→ 每個接頭的內力（6E 維）。
  - 計算方法沿用 `analysis/joint_loads.py` 的框架分析。
  - 每步只要一次矩陣乘法。
- 每步輸出 joint_load_ratio = max over joints 和 {拉力, 剪力, 力矩} 的 (負載 / 上限)。上限來自 module_props.json 的 joint_limits。
- 選項 `detach_on_overload`：預設關閉。開啟時，負載比 > 1 就終止，info 裡記錄原因。

## 介面
```python
env = ModularEnv(cells, n_envs, task="T2", device="cpu", seed=0, perturb=None, cfg=None)
obs = env.reset()                       # [B, N, obs_dim]
obs, reward, terminated, truncated, info = env.step(action)   # reward [B]; terminated/truncated [B]
# info["final_obs"]：在自動 reset 之前，結束那一步的 obs（SAME_STEP 語意，和 ppo.py 的約定相同）
# info["joint_load_ratio"] [B]、info["state"]：完整狀態，給古典控制器和 critic 用
env.adjacency    # [N, N, 4]
env.nominal      # 標稱的 assemble() 結果
```
- reward 和 done 是整隊共用的，形狀 [B]。給 ppo.py 時，由呼叫端 expand 成 [B, N]，見 ppo_interface v0.1。

## 驗收
1. 和 MuJoCo 參考解比對：給相同的隨機推力序列，跑 1 s，位置誤差 < 1e-3 m，姿態誤差 < 1e-3 rad。line3、L3、grid6 各測一次。
2. hover 平衡：用 A 的 pinv 解出 hover 推力，跑 5 s，漂移 < 1e-6 m。
3. 加速度計：靜止 hover 時讀數是 +g ẑ；純 yaw 旋轉時，向心項和手算一致。
4. 接頭內力：靜定構型下，和 `analysis/joint_loads.py` 的瞬間結果一致。
5. 速度：B = 1024、N = 4 時，報告 env step 的 sps，目標 ≥ 20 萬 module-steps/s，未達標就報告實際值。
