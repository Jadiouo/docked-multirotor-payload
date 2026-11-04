# results/media 影片說明

重跑：`CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python scripts/render_videos.py [--only gap event failure] [--resim]`
（failure.gif 本身由 `scripts/render_failure.py` 產生；`render_videos.py` 只把它轉成 MP4。）

所有影片用的環境與控制器參數與論文實驗相同（g2_estim_tune.json best + g1_coord_tune.json best、estim 一律 snap=False、
偵測超參數 rqd_tune.json best），confirmatory seed（offset 900000，100 episode）。每次渲染都重算 100 episode 分數，
並驗證與 `results/p2/classical_confirm_raw.json` 的逐 episode 分數一致（最大差 < 1e-4）。

## episode 選擇規則（事先寫死，不看畫面）
gap_i = 右手方 − 左手方的分數差（定義見各影片）；取 100 個 gap 的平均 gbar，選 argmin_i |gap_i − gbar|，同分取編號最小者。
左右兩邊用同一個環境 seed／目標（同一個環境槽）。畫面上的分數是該 episode 的整段分數（success x (1 − 位置 RMS / e_ref)），不是畫面當下的值。

| 檔案 | 內容 | 選到的 episode | 該 episode 分數 | 100 episode 平均 |
|---|---|---|---|---|
| `e3_gap_line8.mp4/.gif` | E3 偏心 payload（15%，偏移 (+0.0333, −0.0333, −0.0215) m）、line8、POS_STEP。左 estim-yawlim（snap=False），右 informed-offset-yawlim（知道真實 payload 的特權參考）。gap = informed − estim | 86 | 0.366 vs 0.473（差 +0.107） | 0.385 vs 0.494（差 +0.108），與 classical_confirm.md E3 一致 |
| `attach_event_line6.mp4/.gif` | 飛行中偏心掛上 payload（rqd_classical 的 attach-offcenter：15%、掛點 random、事件時刻 U(1.0, 3.5) s；此 episode 事件在 3.47 s）、line6、HOVER。左 estim-reset-yawlim（CUSUM 偵測 + 重置），右 estim-yawlim（無重置）。gap = reset − 無 reset | 59 | 0.962 vs 0.724（差 +0.238） | 0.962 vs 0.724（差 +0.238），與 classical_confirm.md EV 一致 |
| `failure.mp4/.gif` | E3、line6、POS_STEP：B1-QP 墜機 vs estim-yawlim 穩定（探索 seed，非 confirmatory；挑「B1-QP 失敗且 estim 存活」的第一個 episode，是說明用個案，不是代表性樣本） | — | — | — |

注意：
- 事件影片的事件時刻有標在下方誤差圖（黑色虛線）和 caption；事件當下原點（CoM）平移，位置誤差有一個 cm 級的跳變（env 設計，見 quadsim/env.py）。
- 下圖為位置誤差曲線（灰線 = 整段完整曲線，彩線 = 目前已播放）。
- 動畫是 matplotlib 3D 自製示意（MuJoCo 不參與）；即時速度（25 fps，每 4 個控制步一幀），結尾停 1 秒。
- MP4：H.264、yuv420p、1080x720。GIF 為縮小版（720 px 寬、12 fps、<5 MB）。
