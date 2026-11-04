# Online Payload Estimation Nearly Closes the Centered Case but Not the Off-Center One in Docked Modular Multirotors

[中文版](README.zh-TW.md) | [Paper (PDF)](paper/main.pdf)

Several small quadrotor modules are docked into one rigid body and flown by a single centralized controller. The payload they carry is unknown (mass and center-of-mass offset). We study this in a from-scratch batched rigid-body simulator, mainly with classical control: QP thrust allocation, a yaw-rate limit, and recursive-least-squares payload estimation. A from-scratch PPO serves as an exploratory probe. The study was pre-registered (`docs/prereg.md`), and the classical results were recomputed on held-out confirmatory seeds.

## Main results

- Payload is the most expensive of the three errors studied. On exploration seeds, a 15% payload costs the best classical baseline 14.4% of its own nominal score, against 1.7% for a +/-20% thrust-coefficient error and 0.3% for docking error.
- With a centered payload, the estimator stack is within 0.007 of a payload-informed reference (paired difference, confirmatory seeds).
- With an off-center payload, a gap of 8/11/22% (6/7/8 modules) remains in the position-step task.
- The from-scratch PPO did not meet the pre-registered precondition for the main RL experiment, and an exploratory residual PPO on top of the estimator stack did not close the gap for either of two training seeds.

The full method, statistics and limitations are in the paper.

![Classical results on confirmatory seeds](paper/figs/fig2_classical.png)

## Videos

All videos are MP4 renders from the project's own simulator. Details are in [results/media/README.md](results/media/README.md).

- [failure.mp4](results/media/failure.mp4): line6, off-center payload, position step. B1-QP crashes, the estimator stack flies.
- [e3_gap_line8.mp4](results/media/e3_gap_line8.mp4): the remaining off-center gap on line8, estimator stack vs informed reference.
- [attach_event_line6.mp4](results/media/attach_event_line6.mp4): payload attached in flight on line6, with and without reset on the detected event.
- [rl_residual_line6.mp4](results/media/rl_residual_line6.mp4): residual RL vs estim-yawlim, line6, E3 position step.
- [rl_meanbound_line6.mp4](results/media/rl_meanbound_line6.mp4): from-scratch mean-bound RL vs B1, line6, no payload, T2.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest -q tests/
bash scripts/reproduce.sh          # fast: redraw figures from saved json
bash scripts/reproduce.sh --full   # rerun the classical experiments, takes hours
```

`requirements.txt` pins the CPU environment (`.venv`: CPU torch, cadquery, mujoco, gymnasium, tyro). RL training uses a separate `.venv-gpu` with CUDA torch (2.11, cu128); it has no cadquery, so `test_cad` is skipped there. `reproduce.sh` does not rerun RL; figures read saved evaluation json.

```bash
.venv-gpu/bin/python -m scripts.rl_train --run-name t2_single --configs single:T2 --n-envs 2048 --total-steps 40000000 --device cuda
.venv-gpu/bin/python -m scripts.rl_eval results/p3/abl_single_mix_k1_h4/final.pt --configs single --tasks ATT_STEP POS_STEP T2
```

## Repository layout

| Path | Contents |
|---|---|
| `quadsim/` | Batched rigid-body simulator, assembly, controllers, policy, RL environment wrapper |
| `ppo.py` | From-scratch PPO (interface in `docs/ppo_interface.md`) |
| `scripts/` | Experiment scripts, `reproduce.sh`, `rl_train.py`, `rl_eval.py`, video rendering |
| `tests/` | Unit tests |
| `cad/` | CadQuery model of the module |
| `paper/` | LaTeX source, figures, compiled `main.pdf` |
| `results/` | Saved results: `p1` Hopper check, `p2` classical, `p3` RL, `media` videos |
| `docs/prereg.md` | The frozen pre-registration (tag `prereg-v1.0`) |
| `docs/notes_zh.md` | Detailed project notes in Chinese |
| `third_party/` | CleanRL, used for the Hopper sanity check |

## AI assistance

The code, experiment scripts and paper draft in this repository were written by an AI assistant (Claude, Anthropic) under the author's direction. The author reviewed the work and decided the research direction and all trade-offs. In `docs/prereg.md`, the terms commander, white team and user are role names in this review process: the commander plans and verifies, the white team is an independent reviewer, the user is the author. The internal working logs that `docs/prereg.md` cites are not published because they contain private context. `docs/prereg.md` itself is published exactly as frozen.

## License

MIT, see `LICENSE`. `third_party/cleanrl` is also MIT.

## Citation

```bibtex
@misc{ho2026docked,
  author = {Ho, Hao-Ting},
  title  = {Online Payload Estimation Nearly Closes the Centered Case but Not the Off-Center One in Docked Modular Multirotors},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/Jadiouo/docked-multirotor-payload}
}
```
