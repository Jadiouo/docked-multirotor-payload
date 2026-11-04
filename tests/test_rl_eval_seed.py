"""rl_eval 的 --seed-offset：快取隔離、舊 seed 原始檔隔離、env seed 確實加上 offset。CPU：CUDA_VISIBLE_DEVICES= pytest -q tests/test_rl_eval_seed.py"""
import pytest

import scripts.rl_eval as ev
import scripts.g1_gate as g1
from scripts.g1_gate import load_gains

GAINS = load_gains()
FAKE = dict(score=[0.5, 0.5], success=[1, 1], att_rms=[0.1, 0.1], pos_rms=[0.1, 0.1], conv=[1, 1], jmax=[0, 0])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "CACHE", tmp_path / "cache.json")
    calls, reads = [], []

    def fake_run_cell(cells, method, task, n_eps, gains, **kw):
        calls.append((method, task, g1.SEED[task]))
        return dict(FAKE)
    monkeypatch.setattr(ev, "run_cell", fake_run_cell)
    monkeypatch.setattr(ev, "_read_json", lambda p: reads.append(str(p)) or {"_meta": dict(gains=GAINS.to_dict(), n_eps=2), "payload15|single|B1|T2": dict(FAKE, score=[9, 9])})
    real = type(tmp_path).exists
    monkeypatch.setattr(type(tmp_path), "exists", lambda self: True if self.name in ("g2_raw.json", "g2_informed_raw.json", "rqd_classical_raw.json") else real(self))   # 假裝舊 seed 原始檔存在
    return calls, reads, tmp_path


def test_offset0_uses_g2_raw(env):
    calls, reads, _ = env
    c = ev._g2_cell("payload15", "single", "B1", "T2", 2, GAINS, seed_offset=0)
    assert c["score"] == [9, 9] and reads and not calls


def test_offset_nonzero_never_reads_g2_raw_and_isolates_cache(env):
    calls, reads, tmp = env
    for so in (100, 900000):
        c = ev._g2_cell("payload15", "single", "B1", "T2", 2, GAINS, seed_offset=so)
        assert c["score"] == [0.5, 0.5]
    assert not reads                                      # g2_raw.json 完全沒被讀
    assert [x[2] for x in calls] == [g1.SEED["T2"] + 100, g1.SEED["T2"] + 900000]   # env seed = 原 seed + offset
    assert set(g1.SEED.values()) == {0, 1, 2}             # SEED 已還原
    # 同一格不會命中對方的快取：兩個 offset 各有自己的檔與 key
    assert ev.cache_path(100) != ev.cache_path(900000) != ev.cache_path(0)
    assert ev.ckey("k", 100) != ev.ckey("k", 900000) != ev.ckey("k", 0)
    c0 = ev.cache_load(dict(gains=GAINS.to_dict(), n_eps=2), 0)
    assert len(calls) == 2 and "payload15|single|B1|T2" not in c0
    ev._g2_cell("payload15", "single", "B1", "T2", 2, GAINS, seed_offset=900000)   # 再取一次：命中自己的快取
    assert len(calls) == 2


def test_meta_contains_seed_offset_and_legacy_meta_is_offset0(env):
    _, _, tmp = env
    meta = dict(gains=GAINS.to_dict(), n_eps=2)
    c = ev.cache_load(meta, 900000)
    assert c["_meta"]["seed_offset"] == 900000
    # 舊檔（_meta 沒有 seed_offset）視為 offset 0：保留；用 offset 900000 的 meta 去對舊檔內容不會命中
    import json
    ev.CACHE.write_text(json.dumps({"_meta": meta, "x": 1}))
    assert ev.cache_load(meta, 0)["x"] == 1
    assert "x" not in ev.cache_load(meta, 900000)
    ev.cache_path(900000).write_text(json.dumps({"_meta": dict(meta, seed_offset=0), "x": 1}))   # meta 不符 → 丟棄
    assert "x" not in ev.cache_load(meta, 900000)


def test_qp_cell_skips_g1_raw_with_offset(env, monkeypatch):
    calls, _, _ = env
    ev.qp_cell("single", "T2", 2, GAINS, False, seed_offset=100)
    assert calls and calls[-1][2] == g1.SEED["T2"] + 100


def test_event_offset_recomputes_not_raw(env, monkeypatch):
    calls, reads, _ = env
    import scripts.rqd_classical as rq
    seen = []
    monkeypatch.setattr(rq, "run_ev", lambda cells, m, task, n, g, seed, typ, off, **kw: seen.append(seed) or dict(score=[.1, .1], success=[1, 1], peak=[.1, .1], rec_steps=[10, 10]))
    out = ev.compare_event(dict(score=[.2, .2]), "single", "T2", "attach", "center", 2, GAINS, 900000)
    assert not reads and set(seen) == {rq.SEED_EV["T2"] + 900000} and "estim-yawlim" in out
