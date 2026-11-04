# results/media: video notes

Re-run (project root, CPU only):

```
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python scripts/render_videos.py \
    [--only gap event failure rl_residual_line6 rl_meanbound_line6] [--resim]
```

`failure.gif` is produced by `scripts/render_failure.py`; `render_videos.py` only converts it to MP4. Simulation results are cached in `.scratch/render_videos/` (`--resim` forces a re-run).

All videos are side-by-side 3D animations (matplotlib, not MuJoCo) with a position-error plot underneath (grey = full curve, colour = played so far). Every frame shows the controller name, the time, and the score of the whole episode (success x (1 - position RMS / e_ref)), not the instantaneous value. Playback is real time (25 fps, one frame per 4 control steps) with a 1 s hold at the end. MP4: H.264, yuv420p, 1080x720. GIF: 720 px wide, 12 fps, under 5 MB.

## Episode selection rule (fixed in advance, not based on how the video looks)

For each video, gap_i = score_left_i - score_right_i over the 100 episodes (the sign convention is given per video). Let gbar be the mean of the 100 gaps. The shown episode is argmin_i |gap_i - gbar|; ties go to the smallest episode index. Both sides use the same environment seed and goal (same environment slot). Every run recomputes all 100 episode scores and checks them against published results before rendering (checks listed per video).

## Videos

| File | Seed set | Selected episode | Episode score (left vs right) | 100-episode mean (left vs right) |
|---|---|---|---|---|
| `e3_gap_line8` | confirmatory | 86 | 0.366 vs 0.473 | 0.385 vs 0.494 |
| `attach_event_line6` | confirmatory | 59 | 0.962 vs 0.724 | 0.962 vs 0.724 |
| `rl_residual_line6` | validation (exploratory) | 29 | 0.530 vs 0.540 | 0.554 vs 0.565 |
| `rl_meanbound_line6` | validation (exploratory) | 7 | 0.497 vs 0.522 | 0.304 vs 0.328 |

### `e3_gap_line8.mp4/.gif`
- Content: E3 off-center payload (15% mass, offset (+0.0333, -0.0333, -0.0215) m), line8, POS_STEP. Left: estim-yawlim (B1-QP-estim-yawlim, snap=False). Right: informed-offset-yawlim (privileged reference that knows the true payload).
- Setting: confirmatory seed (offset 900000, 100 episodes); parameters from g2_estim_tune.json best + g1_coord_tune.json best.
- Selection: gap = informed - estim-yawlim; episode 86 has gap +0.107 against a mean gap of +0.108.
- Scores: episode 0.366 vs 0.473; 100-episode mean 0.385 vs 0.494. Verified against `results/p2/classical_confirm_raw.json` (per-episode max difference < 1e-4), consistent with classical_confirm.md (E3).

### `attach_event_line6.mp4/.gif`
- Content: an off-center payload (15%, random attachment point) is attached in flight at a random time U(1.0, 3.5) s (rqd_classical attach-offcenter; here at 3.47 s), line6, HOVER. Left: estim-reset-yawlim (CUSUM change detection plus reset). Right: estim-yawlim (no reset).
- Setting: confirmatory seed (offset 900000, 100 episodes); detection hyper-parameters from rqd_tune.json best. The event time is marked on the error plot (dashed line); at the event the origin (CoM) shifts, which causes a centimetre-scale jump in position error (environment design, see quadsim/env.py).
- Selection: gap = estim-reset - estim-yawlim; episode 59 has gap +0.238 against a mean gap of +0.238.
- Scores: episode 0.962 vs 0.724; 100-episode mean 0.962 vs 0.724. Verified against `results/p2/classical_confirm_raw.json` (per-episode max difference < 1e-4), consistent with classical_confirm.md (EV).

### `failure.mp4/.gif`
- Content: E3, line6, POS_STEP: B1-QP crashes while estim-yawlim stays stable.
- Setting: exploration seed, not confirmatory. This is the first episode where B1-QP fails and estim-yawlim survives; it is an illustrative case, not a representative sample. No gap-based selection, no episode score or mean is shown.

### `rl_residual_line6.mp4/.gif`
- Content: residual RL (u = clip(u_estim-yawlim + 0.2 * delta), `results/p3/res_line6_off_k2_h4/final.pt`, training seed 7) on the left versus estim-yawlim (B1-QP-estim-yawlim, snap=False) on the right. line6, E3 off-center payload (same perturbation as `rl_eval.e3_perturb`), POS_STEP.
- Setting: validation seed (env seed offset 100), 100 episodes, `final.pt`, deterministic policy on CPU via `scripts/rl_eval.py` (`load_policy`, `wrap_residual`, `PackedEnv`). Exploratory result, not confirmatory; one training seed.
- Selection: gap = residual RL - estim-yawlim; episode 29 has gap -0.011 against a mean gap of -0.010.
- Scores: episode 0.530 vs 0.540; 100-episode mean 0.554 vs 0.565 (exact: 0.554287 vs 0.564759). Verified against `results/p3/res_line6_off_k2_h4/eval_val_E3.json` (RL and estim-yawlim means equal to 3 decimals) and per episode (max difference 5e-6) against the rl_eval cache `results/p3/rl_eval_cache_so100.json`.

### `rl_meanbound_line6.mp4/.gif`
- Content: mean-bound RL trained from scratch (`results/p3/line6_nom_mb_k2_h4/final.pt`, training seed 7) on the left versus B1 on the right. line6, no payload, T2 (goal switches every 1.5 s; the error plot jumps at each switch).
- Setting: validation seed (env seed offset 100), 100 episodes, `final.pt`, deterministic policy on CPU via `scripts/rl_eval.py`. Exploratory result, not confirmatory; one training seed. The payload marker is hidden (no payload).
- Selection: gap = RL - B1; episode 7 has gap -0.025 against a mean gap of -0.023.
- Scores: episode 0.497 vs 0.522; 100-episode mean 0.304 vs 0.328 (exact: 0.304438 vs 0.327876). Verified against `results/p3/line6_nom_mb_k2_h4/eval_val.json` (RL and B1 means equal to 3 decimals).

In the two RL videos the RL policy is slightly below its classical counterpart on average, so the selected episodes show a small deficit, not a win.
