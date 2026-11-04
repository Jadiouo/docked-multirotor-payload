"""形狀網路（GNN）policy：ModularActorCritic，實作 ppo.py 的 ActorCritic 介面（act / evaluate / value）。

為什麼 obs 要「打包」：
  ppo_update 把 [T, n_envs, N_max] 攤平成一條獨立樣本再隨機抽 minibatch，一筆樣本 = 一個模組。
  但訊息傳遞需要整架的資料，所以每一筆樣本（每個模組那一列）都帶著整架的 obs、鄰接、active mask，
  再加上「我是第幾個模組」的 one-hot。代價是每個模組的資料重複 N_max 次、GNN 也重算 N_max 次；
  好處是完全不用改 ppo.py。（若 ppo_update 能以 [T, n_envs] 為取樣單位，這個重複就可以省掉，見回報。）

一列（row）的內容，長度 = PackLayout.size：
    [ x（N_max × F）| adj（N_max × N_max × 4）| active（N_max）| self（N_max，one-hot）]
  F = 每個模組的特徵長度：actor 是 H × obs_dim（最近 H 步的 obs，舊 → 新；H=1 就是沒有記憶），
      critic 是 obs_dim + 4（目前 obs ＋ 特權資訊：模組相對 CoM 的位置 r_i/pitch（3）、N/N_max（1））。
  adj[i, j, k] = 1 表示模組 j 是模組 i 在方向 k（+x, −x, +y, −y）的鄰居，和 env.adjacency 同定義。

檔案由上到下：
  1) PackLayout    打包／解包的格式
  2) GraphNet      共享 encoder ＋ k 輪 relational message passing
  3) ModularActorCritic
"""
from dataclasses import dataclass
import torch
import torch.nn as nn

from ppo import layer_init

# env.observe() 的欄位順序（28 維）：e_p(3) v_b(3) att(6) gyro(3) | acc(3) yaw(2) flags(4) u_prev(4)
# 固定的縮放，讓各欄位大致落在 ±1 內（不用 running 統計，所以 checkpoint 之間可直接比較）。
# gyro 在 T1 最大約 3 rad/s；acc 在 hover 時 z 軸 ≈ 9.81 m/s²。
OBS_DIM = 28
OBS_SCALE = [1.0] * 12 + [1 / 3] * 3 + [1 / 10] * 3 + [1.0] * 10
PRIV_DIM = 4


# ============================================================================
# 1) 打包格式
# ============================================================================
@dataclass(frozen=True)
class PackLayout:
    n_max: int
    feat: int      # 每個模組的特徵長度 F

    @property
    def size(self):
        n = self.n_max
        return n * self.feat + n * n * 4 + n + n

    def pack(self, x, adj, active):
        """x [B, n, F]、adj [n, n, 4]（所有環境共用）、active [n] → rows [B, n, size]。"""
        B, n = x.shape[0], self.n_max
        tile = lambda t: t.reshape(1, 1, -1).expand(B, n, -1)         # 所有環境共用的東西：複製到每個環境、每個模組
        return torch.cat([x.reshape(B, 1, -1).expand(B, n, -1),       # 整架的 x：每個模組各帶一份
                          tile(adj.to(x.dtype)), tile(active.to(x.dtype)),
                          torch.eye(n, dtype=x.dtype, device=x.device).expand(B, n, n)], -1)

    def unpack(self, rows):
        """rows [..., size] → x [..., n, F]、adj [..., n, n, 4]、active [..., n]、self_onehot [..., n]。"""
        n, F = self.n_max, self.feat
        lead = rows.shape[:-1]
        x, adj, active, sel = rows.split([n * F, n * n * 4, n, n], dim=-1)
        return x.reshape(*lead, n, F), adj.reshape(*lead, n, n, 4), active, sel


# ============================================================================
# 2) GNN
# ============================================================================
class GraphNet(nn.Module):
    """共享 MLP encoder，再做 k 輪 relational message passing。輸入 x [M, n, F]，輸出每個模組的嵌入 [M, n, hidden]。

    每一輪（4 個方向各自一個權重 W_dir，鄰居的訊息直接加總）：
        m_i  = Σ_dir Σ_j adj[i, j, dir] · W_dir h_j
        h_i ← LayerNorm( h_i + tanh( W_self h_i + m_i ) )          （殘差；LayerNorm 可關）
    k = 0 時沒有任何一輪，只剩 encoder，模組只看得到自己的 obs（obs 裡仍含鄰居旗標）。
    active = 0 的位置（補齊用的假模組）：嵌入每一層都被歸零，且邊只保留 active → active 的，
    所以假模組的 obs 內容、或有人誤把邊接到它身上，都不會影響真模組。
    """

    def __init__(self, in_dim, hidden=64, k=1, layer_norm=True):
        super().__init__()
        self.hidden, self.k = hidden, k
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh())
        self.w_self = nn.ModuleList([layer_init(nn.Linear(hidden, hidden)) for _ in range(k)])
        # 一個 Linear 同時產生 4 個方向的訊息：輸出 [..., 4 × hidden] 切成 (4, hidden)
        self.w_dir = nn.ModuleList([nn.Linear(hidden, 4 * hidden, bias=False) for _ in range(k)])
        for lin in self.w_dir:
            nn.init.orthogonal_(lin.weight, 1.0)               # 沒有 bias，所以不能用 layer_init
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) if layer_norm else nn.Identity() for _ in range(k)])

    def forward(self, x, adj, active):
        M, n = x.shape[:2]
        a = active[..., None]                                                   # [M, n, 1]
        adj = adj * active[:, None, :, None] * active[:, :, None, None]         # 只留真模組之間的邊
        h = self.encoder(x) * a
        for w_self, w_dir, norm in zip(self.w_self, self.w_dir, self.norms):
            z = w_dir(h).reshape(M, n, 4, self.hidden)                          # z[b, j, dir] = W_dir h_j
            msg = torch.einsum("bijk,bjkh->bih", adj, z)                        # 對鄰居 j 加總
            h = norm(h + torch.tanh(w_self(h) + msg)) * a
        return h


# ============================================================================
# 3) ModularActorCritic
# ============================================================================
class ModularActorCritic(nn.Module):
    """Actor：GraphNet → 共享 head → 每個模組 4 個馬達的 Gaussian mean，log_std 與狀態無關。
    Critic：另一個 GraphNet（權重不共享）→ 對真模組做 mean pool → [自己的嵌入, 池化向量] 過 head → 每個模組一個 value。
      reward 是整隊共用的，所以每個模組的 value 目標相同（由呼叫端 expand 成 [B, N]）；
      critic 的輸入多了特權資訊（見檔頭），這是 asymmetric critic，actor 拿不到。
    logprob 對 4 個動作維度加總，每個模組各自一個 ratio（MAPPO 的做法）。

    ppo.py 介面：act / evaluate 吃 actor 的 row（[..., actor_layout.size]），value 吃 critic 的 row。
    結構化的入口 actor_forward / critic_forward 吃拆開的 (x, adj, active)，單元測試與分析用。
    動作不做 squash，由環境端 clip 到 [-1, 1]（和 ppo.py 預設一樣）。
    """

    def __init__(self, n_max, hist=1, obs_dim=OBS_DIM, act_dim=4, hidden=64, k=1, critic_k=None,
                 layer_norm=True, log_std_init=-1.0, hover_action=0.0, head_std=0.01, obs_scale=None):
        super().__init__()
        self.hp = dict(n_max=n_max, hist=hist, obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, k=k,
                       critic_k=critic_k, layer_norm=layer_norm, log_std_init=log_std_init,
                       hover_action=hover_action, head_std=head_std)     # 存進 checkpoint，重建 policy 用
        critic_k = k if critic_k is None else critic_k
        self.n_max, self.hist, self.obs_dim = n_max, hist, obs_dim
        self.actor_layout = PackLayout(n_max, hist * obs_dim)
        self.critic_layout = PackLayout(n_max, obs_dim + PRIV_DIM)

        # obs_dim = OBS_DIM + k（殘差 RL 多接 u_base）時，額外欄位縮放 1；obs_dim = OBS_DIM 時與舊版相同
        scale = torch.tensor(OBS_SCALE + [1.0] * (obs_dim - len(OBS_SCALE)) if obs_scale is None else obs_scale, dtype=torch.float32)
        assert scale.numel() == obs_dim
        self.register_buffer("actor_scale", scale.repeat(hist))                                  # H 步各自縮放
        self.register_buffer("critic_scale", torch.cat([scale, torch.ones(PRIV_DIM)]))

        self.actor_net = GraphNet(hist * obs_dim, hidden, k, layer_norm)
        self.actor_head = nn.Sequential(layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                                        layer_init(nn.Linear(hidden, act_dim), std=head_std))   # 初始動作 ≈ hover_action
        # hover_action：初始 mean 設在 hover 附近的動作，避免一開始就大幅偏離 hover 而墜機
        self.register_buffer("action_bias", torch.full((act_dim,), float(hover_action)))
        self.actor_logstd = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        self.critic_net = GraphNet(obs_dim + PRIV_DIM, hidden, critic_k, layer_norm)
        self.critic_head = nn.Sequential(layer_init(nn.Linear(2 * hidden, hidden)), nn.Tanh(),
                                         layer_init(nn.Linear(hidden, 1), std=1.0))

    # ------------------------------------------------------------------ 結構化入口
    @staticmethod
    def _adj(adj, M, n):
        """adj 可以是 [n, n, 4]（所有樣本共用）或 [M, n, n, 4]。"""
        return adj.expand(M, n, n, 4) if adj.dim() == 3 else adj

    def actor_forward(self, x, adj, active):
        """x [B, N, H·obs_dim]、adj [N, N, 4] 或 [B, N, N, 4]、active [B, N] → 每個模組的 mean [B, N, act_dim]。
        假模組（active = 0）的輸出固定為 0。"""
        M, n = x.shape[:2]
        h = self.actor_net(x * self.actor_scale, self._adj(adj, M, n), active)
        return (self.actor_head(h) + self.action_bias) * active[..., None]

    def critic_forward(self, x, adj, active):
        """x [B, N, obs_dim + 4]（含特權資訊）→ 每個模組的 value [B, N]（正規化尺度，見 ppo_update 的 value_rms 合約）。"""
        M, n = x.shape[:2]
        h = self.critic_net(x * self.critic_scale, self._adj(adj, M, n), active)
        pooled = (h * active[..., None]).sum(1, keepdim=True) / active.sum(1).clamp(min=1)[:, None, None]
        return self.critic_head(torch.cat([h, pooled.expand_as(h)], -1)).squeeze(-1) * active

    # ------------------------------------------------------------------ ppo.py 介面（吃打包後的 row）
    def _flat_rows(self, rows, layout):
        """[..., size] → 攤平前導維度 → x, adj, active, sel（第 0 維是樣本）。"""
        x, adj, active, sel = layout.unpack(rows.reshape(-1, rows.shape[-1]))
        return x, adj, active, sel

    def _mean(self, obs):
        lead = obs.shape[:-1]
        x, adj, active, sel = self._flat_rows(obs, self.actor_layout)
        mean_all = self.actor_forward(x, adj, active)                 # 整架的 mean [M, n, act_dim]
        mean = (mean_all * sel[..., None]).sum(1)                     # 挑出「我」這個模組
        return mean.reshape(*lead, -1)

    def _dist(self, obs):
        mean = self._mean(obs)
        return torch.distributions.Normal(mean, self.actor_logstd.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """rollout 用。回傳 (action [..., act_dim], logprob [...])，logprob 為各動作維度總和。"""
        dist = self._dist(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1)

    def evaluate(self, obs, action):
        dist = self._dist(obs)
        self.last_mean = dist.mean                                    # 供 ppo_loss 的 mean_bound_coef 使用（pre-clip 均值，帶梯度）
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    def value(self, critic_obs):
        lead = critic_obs.shape[:-1]
        x, adj, active, sel = self._flat_rows(critic_obs, self.critic_layout)
        v_all = self.critic_forward(x, adj, active)                   # [M, n]
        return (v_all * sel).sum(1).reshape(lead)

    # ------------------------------------------------------------------ rollout 用的「整隊共用」入口
    # rollout 時同一個環境的 n_max 個模組那幾列，x / adj / active 完全相同（只有 self one-hot 不同），
    # 所以整架的 GNN 只需要算一次，再把每個模組的結果挑出來。數值與 act / value 逐列呼叫完全一致（見 test_policy），
    # 但 GNN 的計算量少 n_max 倍。ppo_update 的 minibatch 是隨機抽「單列」，沒辦法共用，那部分的重複仍在（要改 ppo.py）。
    @torch.no_grad()
    def act_team(self, obs, deterministic=False):
        """obs [E, n_max, actor_size]（每個環境的 n_max 列）→ (action [E, n_max, act_dim], logprob [E, n_max])。"""
        x, adj, active, _ = self.actor_layout.unpack(obs[:, 0])        # 每個環境取第 0 列：整架資料都在裡面
        mean = self.actor_forward(x, adj, active)                      # [E, n, act_dim]，假模組為 0
        dist = torch.distributions.Normal(mean, self.actor_logstd.exp().expand_as(mean))
        action = mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def value_team(self, critic_obs):
        """critic_obs [E, n_max, critic_size] → value [E, n_max]（正規化尺度，每個模組一個，假模組為 0）。"""
        x, adj, active, _ = self.critic_layout.unpack(critic_obs[:, 0])
        return self.critic_forward(x, adj, active)

    # ------------------------------------------------------------------ 整隊樣本（--team-sample）的入口
    # 樣本單位 = 一個環境（一整隊）。輸入只要每個環境的「一列」（x / adj / active 整隊相同），不必帶 n_max 份。
    def team_logprob_entropy(self, obs_row, actions):
        """obs_row [M, actor_size]、actions [M, n_max·act_dim]（攤平）→ (logprob [M], entropy [M], mean [M, n, act_dim], active [M, n])。
        logprob / entropy 對「有效模組」（active=1）與動作維度加總（聯合分佈）；假模組用 where 排除，其內容不會影響結果。"""
        x, adj, active, _ = self.actor_layout.unpack(obs_row)
        mean = self.actor_forward(x, adj, active)
        a = actions.reshape(*mean.shape)
        dist = torch.distributions.Normal(mean, self.actor_logstd.exp().expand_as(mean))
        m = active > 0
        zero = torch.zeros((), device=mean.device)
        lp = torch.where(m, dist.log_prob(a).sum(-1), zero).sum(-1)
        ent = torch.where(m, dist.entropy().sum(-1), zero).sum(-1)
        return lp, ent, mean, active

    def team_value(self, critic_row):
        """critic_row [M, critic_size] → 每隊一個 value [M]（帶梯度、正規化尺度）= 有效模組 value 的平均。"""
        x, adj, active, _ = self.critic_layout.unpack(critic_row)
        v = self.critic_forward(x, adj, active)
        return v.sum(-1) / active.sum(-1).clamp(min=1)
