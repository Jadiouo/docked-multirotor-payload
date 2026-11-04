# ppo.py 介面需求（尚未實作）

來源：04 ctde-mappo-vmas session，2026-10-24。這些需求也正好符合本專案自己的需要：模組化多旋翼的每個模組都跑同一份 policy，屬於多 agent 架構。

1. 張量形狀要允許多一個 agent 維度，例如 `[T, n_envs, n_agents, ...]`。GAE 要寫成純函式：輸入 `(rewards, values, dones, truncs)`，輸出 `(advantages, returns)`，不寫死前面有幾個維度。minibatch 切分也不能寫死。
2. buffer 裡把 actor 的輸入 `obs` 和 critic 的輸入 `critic_obs` 分成兩個欄位。單機時 `critic_obs = obs`。這是 CTDE 的核心。
3. value normalization（running mean/std 或 PopArt）寫成獨立的小模組，給 value loss 和 GAE 共用。
4. `terminated` 和 `truncated` 分開存，遇到 time limit 時要 bootstrap。
5. policy 是 `torch.nn.Module`，提供兩個方法：
   - `act(obs) -> action, logprob`
   - `evaluate(obs, action) -> logprob, entropy`
6. 只用 CPU 也要能跑，device 由參數傳入，不寫死 cuda。

狀態：介面文件寫好之後通知 ctde-mappo-vmas-08。
