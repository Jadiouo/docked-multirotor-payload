# ppo.py 介面規格（v0.2：已實作，2026-10-25）

## v0.2 實作時新增的介面
- `RolloutBuffer.add(t, obs, critic_obs, action, logprob, reward, value, terminated, truncated, next_value, active=None)`：寫入第 t 步。
- `RolloutBuffer.compute_advantages(gamma, lam)`：呼叫 `compute_gae`，結果存進 `buffer.advantages`（未正規化）和 `buffer.returns`。
- `ppo_loss(agent, batch, cfg, value_rms=None)`：單一 minibatch 的 loss，`ppo_update` 內部使用，也方便測試。
- `ppo_update(agent, optimizer, buffer, cfg, value_rms=None)`：`value_rms` 可選。給了就是 critic 輸出正規化尺度、buffer 的 values／next_values 為真實尺度；rms 在更新開頭用 active 的 returns 刷新，value clip 的錨點會稍微偏移，沒有 PopArt 補償（已知限制）。
- `PPOConfig.adv_norm_scope`：`"minibatch"`（預設，同 CleanRL）或 `"buffer"`（整個 buffer 依 mask 正規化一次，給 active 很稀疏的情況用）。
- `mask`（`active`）必須是 0/1；masked 平均用 `torch.where`，padding 位置的 NaN／inf 不會污染結果。

以下是原本的 v0 草案文字：

## 設計原則
- 單一檔案，可以從頭讀到尾。只依賴 torch 和 numpy，不依賴任何 RL 函式庫。
- 張量的前導維度不寫死。rollout 的形狀是 `[T, *B]`，其中 `B` 可以是 `(n_envs,)`，也可以是 `(n_envs, n_agents)`。
  - GAE 和 minibatch 都把 `*B` 攤平處理，所以多一個 agent 維度時程式不用改。
- device 由參數傳入，預設是 `"cpu"`。

## 元件
```python
# 1) GAE：純函式。輸入的時間軸是第 0 維，其餘維度任意。
def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """
    rewards, values, terminated, truncated: [T, *B]
    next_values: [T, *B]，是 V(s_{t+1})
      - terminated 的那一步：next_value 視為 0
      - truncated（撞到 time limit）的那一步：用真正的 s_{t+1} 做 bootstrap，
        GAE 的鏈在這裡斷開，不往前一個 episode 傳
    回傳 advantages, returns，形狀都是 [T, *B]
    """

# 2) value 正規化：獨立的小模組，value loss 和 GAE 都會用到
class RunningMeanStd:          # 用來正規化 obs、return、value
    def update(self, x): ...
    def normalize(self, x): ...
    def denormalize(self, x): ...

# 3) policy 介面：任何 nn.Module，只要實作下面三個方法就能換掉
class ActorCritic(nn.Module):
    def act(self, obs) -> tuple[action, logprob]          # rollout 時使用，不需要梯度
    def evaluate(self, obs, action) -> tuple[logprob, entropy]
    def value(self, critic_obs) -> value                  # critic 吃 critic_obs，不吃 obs

# 4) rollout buffer：obs 和 critic_obs 分成兩個欄位
#    欄位：obs, critic_obs, actions, logprobs, rewards, values,
#          terminated, truncated, next_values
#    單機時 critic_obs = obs

# 5) 更新：給定 buffer，做 n_epochs × n_minibatches 次 PPO 更新
#    損失 = clipped surrogate + value loss（可選擇是否 clip）- entropy bonus
def ppo_update(agent, optimizer, buffer, cfg) -> dict   # 回傳要記錄的統計量
```

## v0.1 補充（2026-10-24，依 04 的意見全部採納）
1. **mask 提前到 P1**：buffer 加一個可選欄位 `active`，形狀 `[T, *B]`，預設全為 1。下面這幾項都改成依 mask 加權平均：policy loss、value loss、entropy，以及 advantage 正規化的平均數和標準差。本專案補齊到 N_max 時也用這個欄位。
2. **done 的形狀由呼叫端先 expand**：`compute_gae` 不自動廣播。`rewards`、`values`、`terminated`、`truncated`、`next_values` 的形狀必須完全相同；如果整隊共用一個 done（形狀是 `[T, n_envs]`），呼叫端要自己 expand 到 `[T, n_envs, n_agents]`。選這個做法是因為明確比較不容易出錯，函式裡會用 assert 檢查形狀。
3. **next_values 由呼叫端計算**，而且要用 **reset 之前的 final obs**（gymnasium 的 `final_obs`）算。autoreset 回傳的是新 episode 的 obs，不能拿來 bootstrap。
4. `act(obs, deterministic=False)`：`deterministic=True` 時回傳 Gaussian 的 mean。
5. **動作界線可以替換**：分佈和 squash 方式（不處理、clip、tanh 擇一）寫在 `ActorCritic` 子類裡，`ppo.py` 的 update 只用 `evaluate()` 回傳的 logprob 和 entropy，不寫死。預設用「Gaussian 加上在環境端 clip」，和 CleanRL 一樣。
6. `ppo_update` 回傳的統計量至少包含 `approx_kl`、`clip_frac`、`explained_variance`、`grad_norm`，以及 policy／value／entropy 三項 loss。buffer 保留 `advantages` 欄位（未正規化的版本），供外部分析。
7. `cfg` 包含 `max_grad_norm`、`clip_vloss`（bool）、`clip_coef`、`vf_coef`、`ent_coef`、`n_epochs`、`n_minibatches`、`norm_adv`。

## 給 04（MAPPO）的延伸點
- 參數共享加上 agent ID：自己寫一個 `ActorCritic` 子類，把 agent one-hot 接在 obs 後面。
- 集中式 critic：`critic_obs` 放全域 state。buffer 不用改。
- 在 `[T, n_envs, n_agents]` 的形狀下，GAE 和 update 都能直接用。

## 本專案自己的用法（模組化多旋翼）
- 每個模組就是一個 agent，所以 `B = (n_envs, n_modules)`。
- 不同構型的 n_modules 不同。處理方式是補齊到 N_max，再用 mask 遮掉沒有模組的位置（mask 欄位在 P3 才加，加的時候會通知 04）。
