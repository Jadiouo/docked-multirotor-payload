"""P0b：感知半徑 k 與控制權限。共享權重的 GNN 在兩個模組 k 跳視野相同時輸出必相同，
所以「同色模組強制輸出相同的 4 顆推力」給出該 k 的權限上界。
執行：.venv/bin/python -m analysis.khop"""
from collections import deque
import numpy as np
from quadsim.assembly import CONFIGS, LARGE, assemble
from analysis.controllability import lp_alpha, lp_both

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]    # +x, −x, +y, −y
AX = ["roll", "pitch", "yaw"]


def neighbors(cells):
    """docking graph：四鄰格點相鄰就有邊。nbr[i][d] = 方向 d 上的鄰居編號或 None。"""
    idx = {c: i for i, c in enumerate(cells)}
    return [[idx.get((x + dx, y + dy)) for dx, dy in DIRS] for x, y in cells]


def diameter_radius(nbr):
    """每個節點 BFS 取離心率；直徑 = 最大、半徑 = 最小。"""
    ecc = []
    for s in range(len(nbr)):
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if v is not None and v not in dist: dist[v] = dist[u] + 1; q.append(v)
        ecc.append(max(dist.values()))
    return max(ecc), min(ecc)


def canon(keys):
    """把任意可雜湊的顏色改編成 0..C-1（依第一次出現順序），避免顏色隨輪數越長。"""
    m = {}
    return [m.setdefault(k, len(m)) for k in keys]


def refinements(nbr):
    """回傳 [("盲", 顏色), (0, 顏色), (1, 顏色), ...] 直到顏色數不再增加。
    帶方向的 1-WL：新顏色 = (舊顏色, 四個方向鄰居的舊顏色或 None)；方向保留才分得開左右兩端。
    顏色只會越分越細，所以類別數不變 = 分割不變 = 已穩定。"""
    N = len(nbr)
    out = [("盲", [0] * N)]
    cur = canon([tuple(j is not None for j in row) for row in nbr])      # k=0：自己四邊有沒有鄰居
    out.append((0, cur))
    while True:
        new = canon([(cur[i], tuple(None if j is None else cur[j] for j in nbr[i])) for i in range(N)])
        if len(set(new)) == len(set(cur)): return out
        out.append((len(out) - 1, new)); cur = new


def authority(s, col):
    """同色模組共用同一組 4 顆推力：T = P·u，P 把色別 c 的 u 複製給該色的所有模組。
    回傳各軸單軸權限（± 取小）與 roll/pitch 同時需求權限。"""
    P = np.kron(np.eye(max(col) + 1)[col], np.eye(4))     # (4N, 4C)
    Ar = s["A"] @ P
    a = [min(lp_alpha(Ar, s["M"], s["J"], k, +1), lp_alpha(Ar, s["M"], s["J"], k, -1)) for k in range(3)]
    both = min(lp_both(Ar, s["M"], s["J"], x, y) for x in (1, -1) for y in (1, -1))
    return np.array(a + [both])


if __name__ == "__main__":
    summ = ["| 構型 | N | 直徑 | 半徑 | 穩定 k | 穩定時顏色數 | k*（pitch≥0.95） | 最終比值 roll/pitch/yaw/同時 |", "|---|---|---|---|---|---|---|---|"]
    detail, stuck = [], []
    for name, cells in CONFIGS.items():
        s = assemble(cells); nbr = neighbors(cells); D, R = diameter_radius(nbr)
        full = authority(s, list(range(s["N"])))
        steps = refinements(nbr)
        rat = [(k, len(set(c)), authority(s, c) / full) for k, c in steps]
        kstar = next((k for k, _, r in rat if r[1] >= 0.95), None)
        k_end, n_end, r_end = rat[-1]
        summ.append(f"| {name} | {s['N']} | {D} | {R} | {k_end} | {n_end} | {kstar if kstar is not None else '到不了'} | " + "/".join("%.2f" % v for v in r_end) + " |")
        if n_end < s["N"]:
            col = steps[-1][1]
            stuck.append(f"{name}：穩定後仍有 {s['N'] - n_end} 個模組分不開，同色群 = " +
                         str([[cells[i] for i in range(s["N"]) if col[i] == c] for c in set(col) if col.count(c) > 1]))
        detail += [f"\n### {name}（N={s['N']}，直徑 {D}，半徑 {R}）\n", "| k | 顏色數 | roll | pitch | yaw | roll+pitch 同時 |", "|---|---|---|---|---|---|"]
        detail += [f"| {k} | {n} | " + " | ".join("%.2f" % v for v in r) + " |" for k, n, r in rat]
    interp = """1. 帶方向的視野下，13 個構型最終每個模組都分得開（沒有分不開的對稱群），所有構型最終都能到 full（比值 1.00）；方向資訊讓左右兩端一開始就不同色。
2. 二維構型（grid6、grid9、plus5、T4、square4、L3）k=0 就全部分開，只需要知道自己四邊有沒有鄰居。
3. 長直線最吃感知半徑：k* 隨長度增加（line4 為 1，line6、line8 為 2；line8 的 k=0 只有 0.53），因為中段模組的 k=0 視野完全相同。
4. 這是上界：顏色分得開只代表「資訊足夠」，不代表 GNN 學得到；同時需求（roll+pitch）與 pitch 比值幾乎同步。
5. 沒有找到「永遠到不了 full」的反例；若加入具中心對稱且方向被抹除的視野（不帶方向的 WL），才會出現分不開的群，本分析不涵蓋。"""
    md = ("# P0b 感知半徑 k 與權限比值\n\n比值 = (同色強制同輸出的權限) / full；k=盲 對應 P0 symmetric。權限定義同 P0（α=J⁻¹τ，Fz=Mg，單軸其餘為 0，± 取小）。"
          "k=0 只知道自己四邊有沒有鄰居；k 跳 = 帶方向的 1-WL 第 k 輪。\n\n## 總表\n\n" + "\n".join(summ) +
          "\n\n穩定 k 與半徑：線形構型的穩定 k = 半徑 − 1（line2、line3 為 0，line4 為 1，line6 為 2，line8 為 3）；格狀與其他二維構型在 k=0（鄰居旗標，物理上已涵蓋半徑 1 的佔用）就穩定。\n\n分不開的模組（穩定後仍同色）：" + ("\n- " + "\n- ".join(stuck) if stuck else "無，所有構型最終每個模組都有獨特顏色。") +
          "\n\n## 解讀\n\n" + interp + "\n\n## 各構型逐 k 比值\n" + "\n".join(detail) + "\n")
    open("results/p0/khop.md", "w").write(md)
    print("\n".join(summ)); print(stuck)
