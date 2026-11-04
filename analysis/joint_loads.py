"""P0d：接頭內力的解析分析。驗證主張：「均勻質量＋pseudoinverse 分配下，模組 i 的推力增量 ∝ r_i，
正好抵消它的慣性力，所以 oracle 的接頭內力趨近 0；yaw 反扭矩每個模組一樣但慣性需求 ∝ r，所以 yaw 對任何分配都有面內剪力」。

模型（瞬間狀態，不積分；座標 x 前、y 左、z 上，所有模組朝向相同、機體水平）
1. 整架是剛體。給定 4N 顆推力 T（先 clip 到 [0, Tmax]）與角速度 ω：
   M a_c = (ΣT − M g) ẑ；α = J⁻¹(τ − ω×Jω)，τ = A[1:]·T（對 CoM）。
2. 模組 i 的 d'Alembert 外載（參考點 = 模組中心 c_i，d_i = c_i − CoM）：
   力  F_i = (自己 4 顆推力和 − m_i g) ẑ − m_i (a_c + α×d_i + ω×(ω×d_i))
   矩  M_i = 自己推力對模組中心的力矩 (y_l T, −x_l T) + yaw 反扭矩 (km·spin·T) − J0 α − ω×J0ω
   總和必為 0：ΣF_i = 0 且 Σ(M_i + d_i×F_i) = 0（程式內 assert）。
3. 接頭：docking graph 上每條邊（格點距離 1）一個接頭，位在兩中心中點。
   樹狀構型（single、line*、L3、T4、plus5）= 靜定：截面法（自葉往根累加子樹外載，唯一解，與勁度無關）。
   含迴路（square4、grid6、grid9、grid10）= 靜不定：剛性模組＋6 自由度線彈性接頭，勁度 diag(kt,kt,kt,kr,kr,kr)，
   小位移解 BᵀKB u = ext（lstsq 吃掉 6 個剛體模態），接頭內力 s = K B u。解出的結果取決於 kr/kt（見報告）。
   回報量 = 接頭施加在較大編號模組上的力與力矩（作用在中點），投影到接頭座標（n = 兩中心連線，t = ẑ×n，ẑ）：
   拉力/壓力 = |f·n|；面內剪力 = |f·t|；面外剪力 = |f·ẑ|；彎矩 = hypot(m·ẑ, m·t)；扭矩 = |m·n|。
4. 分配方法（指令 = 目標角加速度 α_cmd；Fz = M g）
   oracle：T = pinv(A)·[M g, J α_cmd]（hover 加 wrench 的最小範數解）。
   sym：每模組同樣 4 顆推力，u = pinv(sym_matrix)·同一個 wrench。
   古典 k 跳（k=0,1）：模組 i 的球 B_i = docking graph 上 (k+1) 跳內的模組（k=0 = 自己加直接鄰居；單機時球只有自己）。
     ĉ = 球內中心平均；r̂ = 球內各轉子相對 ĉ 的位置；Ĵ = Σ_球內 [J0 + m(|r̂_c|²I − r̂_c r̂_cᵀ)]（r̂_c = 模組中心相對 ĉ）。
     需要的力矩 τ̂ = Ĵ α_cmd。roll/pitch：球內 pseudoinverse 形式 ΔT = P⁺ τ̂_xy，P = [y; −x]（球內轉子，相對 ĉ），
     模組 i 只取自己 4 顆。yaw：τ̂_z 在球內均分，每顆轉子 spin·τ̂_z/(n_球·4·km)。Fz：每顆 m g/4。
     球 = 全架時等於 oracle（測試涵蓋）。
5. 機動：hover；pitch/roll/yaw 最大角加速度（oracle 的最小範數解不 clip 時、所有推力仍在 [0,Tmax] 的最大 |α|，
   取正負方向較小者，施加正向）的 50%、100%（pitch）／100%（roll、yaw）；payload = 質點，質量 = 30% 的「含 payload 總質量」
   （m_p = 0.3/0.7·M），掛在 line 構型 x 最大的模組中心，oracle 用含 payload 的 CoM、J 重算。
   所有方法都 clip；達成率 = 實得 α / 指令 α。
執行（專案根目錄）：.venv/bin/python -m analysis.joint_loads"""
import numpy as np
from quadsim.params import CF2X, G, MODULE_PITCH
from quadsim.assembly import CONFIGS, MOTOR_XY, MOTOR_SPIN

KT, KR = 1e4, 25.0            # N/m、N·m/rad：kr = kt·(0.05 m)²，使平移與旋轉對接頭變形的貢獻同量級
PAYLOAD_FRAC = 0.3            # payload / (模組總質量 + payload)
Z = np.array([0., 0., 1.])
EDGES_CACHE = {}


def build(cells, payload=0.0, host=None, p=CF2X, pitch=MODULE_PITCH):
    """整架模型。payload = 質點質量（kg），掛在模組 host 的中心。
    p、pitch 預設是 P0 的 CF2X 單機參數；P2 環境傳入 CAD 模組參數（quadsim/module_params.py）。"""
    m = p["mass"]
    c = np.c_[np.asarray(cells, float) * pitch, np.zeros(len(cells))]
    N = len(c); mass = np.full(N, m)
    if payload: mass[host] += payload
    M = mass.sum(); com = (mass[:, None] * c).sum(0) / M
    d = c - com
    J0 = np.diag([p["Ixx"], p["Iyy"], p["Izz"]])
    J = sum(J0 + mi * (di @ di * np.eye(3) - np.outer(di, di)) for mi, di in zip(mass, d))
    pos = np.vstack([di + np.r_[xy, 0] for di in d for xy in MOTOR_XY])
    spin = np.tile(MOTOR_SPIN, N)
    A = np.vstack([np.ones(4 * N), pos[:, 1], -pos[:, 0], p["km_ratio"] * spin])
    return dict(cells=list(cells), N=N, c=c, d=d, mass=mass, M=M, com=com, J=J, J0=J0, pos=pos, spin=spin, A=A, m=m, p=p, pitch=pitch)


def edges(cells):
    """(i, j) i<j，格點距離 1。"""
    idx = {c: i for i, c in enumerate(cells)}
    return sorted(tuple(sorted((idx[c], idx[(c[0] + dx, c[1] + dy)]))) for c in cells for dx, dy in ((1, 0), (0, 1)) if (c[0] + dx, c[1] + dy) in idx)


def is_tree(mod): return len(edges(mod["cells"])) == mod["N"] - 1


def rigid_body(mod, T, omega=np.zeros(3)):
    """回傳 a_c, alpha（T 已 clip）。"""
    tau = mod["A"][1:] @ T; J = mod["J"]
    alpha = np.linalg.solve(J, tau - np.cross(omega, J @ omega))
    return np.array([0, 0, T.sum() / mod["M"] - G]), alpha


def module_loads(mod, T, omega=np.zeros(3)):
    """每個模組的 d'Alembert 外載 (N,6)=[F, M]（參考點 = 模組中心），含總和為 0 的 assert。"""
    T = np.asarray(T, float); a_c, alpha = rigid_body(mod, T, omega)
    ext = np.zeros((mod["N"], 6)); km = mod["p"]["km_ratio"]
    for i in range(mod["N"]):
        Ti = T[4 * i:4 * i + 4]; mi = mod["mass"][i]; di = mod["d"][i]
        a_i = a_c + np.cross(alpha, di) + np.cross(omega, np.cross(omega, di))
        F = np.array([0, 0, Ti.sum()]) - mi * G * Z - mi * a_i
        M = np.array([MOTOR_XY[:, 1] @ Ti, -MOTOR_XY[:, 0] @ Ti, km * MOTOR_SPIN @ Ti]) \
            - mod["J0"] @ alpha - np.cross(omega, mod["J0"] @ omega)
        ext[i] = np.r_[F, M]
    scale = max(mod["M"] * G, 1e-12)
    tot_F = ext[:, :3].sum(0)
    tot_M = sum(e[3:] + np.cross(di, e[:3]) for e, di in zip(ext, mod["d"]))
    assert np.abs(tot_F).max() < 1e-9 * scale, tot_F
    assert np.abs(tot_M).max() < 1e-9 * scale * mod["pitch"] * mod["N"], tot_M
    return ext


def transfer(w, P, Q):
    """把 (F, M about P) 換成 about Q。"""
    return np.r_[w[:3], w[3:] + np.cross(P - Q, w[:3])]


def joints_tree(mod, ext):
    """截面法。回傳 {(i,j): 接頭對模組 j（較大編號）的 [f, m]，全域座標，作用於中點}。"""
    E = edges(mod["cells"]); N = mod["N"]; c = mod["c"]
    adj = {i: [] for i in range(N)}
    for i, j in E: adj[i].append(j); adj[j].append(i)
    parent, order = {0: None}, [0]
    for u in order:
        for v in adj[u]:
            if v not in parent: parent[v] = u; order.append(v)
    S = {k: ext[k].copy() for k in range(N)}         # 子樹外載，參考點 c_k
    out = {}
    for k in reversed(order):
        q = parent[k]
        if q is None: continue
        mid = (c[k] + c[q]) / 2
        on_child = -transfer(S[k], c[k], mid)        # 接頭對子樹的力 = −子樹外載（平衡）
        out[(min(k, q), max(k, q))] = on_child if k > q else -on_child
        S[q] = S[q] + transfer(S[k], c[k], c[q])
    return out


def joints_frame(mod, ext, kt=KT, kr=KR, check=True):
    """框架小位移分析（一般構型）。check=False 時不檢查平衡殘差（建線性映射時，單位基底不是自平衡的外載）。"""
    E = edges(mod["cells"]); N = mod["N"]; c = mod["c"]
    if not E: return {}
    B = np.zeros((6 * len(E), 6 * N)); K = np.diag(np.tile([kt] * 3 + [kr] * 3, len(E)))
    for e, (i, j) in enumerate(E):
        mid = (c[i] + c[j]) / 2
        for k, sgn in ((j, 1.), (i, -1.)):
            r = mid - c[k]; rx = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            B[6 * e:6 * e + 3, 6 * k:6 * k + 3] = sgn * np.eye(3)
            B[6 * e:6 * e + 3, 6 * k + 3:6 * k + 6] = -sgn * rx
            B[6 * e + 3:6 * e + 6, 6 * k + 3:6 * k + 6] = sgn * np.eye(3)
    f = ext.reshape(-1)
    # 對角縮放改善條件數（平移 ↔ 公尺、旋轉 ↔ 弧度）
    u = np.linalg.lstsq(B.T @ K @ B, f, rcond=1e-12)[0]
    if check: assert np.abs(B.T @ K @ B @ u - f).max() < 1e-6 * max(np.abs(f).max(), 1e-6)
    s = (K @ B @ u).reshape(-1, 6)
    return {e: -s[n] for n, e in enumerate(E)}      # 接頭對 j 的力 = −s


def joints(mod, ext, method="auto", kt=KT, kr=KR):
    if method == "auto": method = "tree" if is_tree(mod) else "frame"
    return joints_tree(mod, ext) if method == "tree" else joints_frame(mod, ext, kt, kr)


def project(mod, W):
    """每個接頭 → dict(axial, shear_ip, shear_oop, bend, torsion)（皆取絕對值）。"""
    rows = []
    for (i, j), w in W.items():
        n = mod["c"][j] - mod["c"][i]; n = n / np.linalg.norm(n); t = np.cross(Z, n)
        f, m = w[:3], w[3:]
        rows.append(dict(edge=(i, j), axial=abs(f @ n), shear_ip=abs(f @ t), shear_oop=abs(f @ Z),
                         bend=np.hypot(m @ Z, m @ t), torsion=abs(m @ n)))
    return rows


def joint_map(mod, method="auto", kt=KT, kr=KR):
    """接頭內力對外載的線性映射（內力與外載成線性，所以對 6N 個單位基底各算一次）。
    輸入：外載 ext 攤平成 (6N,)，順序同 module_loads（每模組 [Fx,Fy,Fz,Mx,My,Mz]，參考點 = 模組中心）。
    輸出：L，形狀 (E, 6, 6N)，邊的順序同 edges(cells)；每條邊 6 個分量在接頭座標：
      [f·n, f·t, f·ẑ, m·n, m·t, m·ẑ]，n = 兩中心連線（i→j，i<j），t = ẑ×n。
    只在「自平衡」的外載（d'Alembert 外載總和為 0）上和 joints() 完全一致。"""
    if method == "auto": method = "tree" if is_tree(mod) else "frame"
    E, N, c = edges(mod["cells"]), mod["N"], mod["c"]
    L = np.zeros((len(E), 6, 6 * N))
    if not E: return L
    frames = []
    for i, j in E:
        n = c[j] - c[i]; n = n / np.linalg.norm(n)
        frames.append(np.stack([n, np.cross(Z, n), Z]))              # 列 = n, t, ẑ
    for k in range(6 * N):
        ext = np.zeros(6 * N); ext[k] = 1.0; ext = ext.reshape(N, 6)
        W = joints_tree(mod, ext) if method == "tree" else joints_frame(mod, ext, kt, kr, check=False)
        for e, (i, j) in enumerate(E):
            w, Fm = W[(i, j)], frames[e]
            L[e, :3, k], L[e, 3:, k] = Fm @ w[:3], Fm @ w[3:]
    return L


KEYS = ["axial", "shear_ip", "shear_oop", "bend", "torsion"]


def max_over_joints(rows):
    return {k: max((r[k] for r in rows), default=0.0) for k in KEYS}


# ---- 分配 ----
def wrench(mod, alpha_cmd, omega=np.zeros(3)):
    return np.r_[mod["M"] * G, mod["J"] @ alpha_cmd + np.cross(omega, mod["J"] @ omega)]


def alloc_oracle(mod, alpha_cmd, omega=np.zeros(3)):
    return np.linalg.pinv(mod["A"]) @ wrench(mod, alpha_cmd, omega)


def alloc_sym(mod, alpha_cmd, omega=np.zeros(3)):
    As = mod["A"] @ np.tile(np.eye(4), (mod["N"], 1))
    return np.tile(np.linalg.pinv(As) @ wrench(mod, alpha_cmd, omega), mod["N"])


def hop_ball(cells, i, R):
    E = edges(cells); adj = {k: set() for k in range(len(cells))}
    for a, b in E: adj[a].add(b); adj[b].add(a)
    dist, fr = {i: 0}, [i]
    for _ in range(R):
        fr = [v for u in fr for v in adj[u] if v not in dist and not dist.update({v: 0})]
    return sorted(dist)


def alloc_khop(mod, alpha_cmd, k, omega=np.zeros(3)):
    p = mod["p"]; km = p["km_ratio"]; m = mod["m"]; cells = mod["cells"]; N = mod["N"]   # 用 build() 帶進來的參數（CAD 或 CF2X），不寫死
    ctr = np.asarray(cells, float) * mod["pitch"]
    ctr = np.c_[ctr, np.zeros(N)]
    T = np.zeros(4 * N)
    for i in range(N):
        ball = hop_ball(cells, i, k + 1); n = len(ball)
        chat = ctr[ball].mean(0); rc = ctr[ball] - chat
        Jh = sum(mod["J0"] + m * (r @ r * np.eye(3) - np.outer(r, r)) for r in rc)
        tau = Jh @ alpha_cmd
        rot = np.vstack([r + np.r_[xy, 0] for r in rc for xy in MOTOR_XY])
        P = np.vstack([rot[:, 1], -rot[:, 0]])
        dT = np.linalg.pinv(P) @ tau[:2]
        s = ball.index(i) * 4
        T[4 * i:4 * i + 4] = m * G / 4 + dT[s:s + 4] + MOTOR_SPIN * tau[2] / (n * 4 * km)
    return T


def allocate_raw(mod, method, alpha_cmd, omega=np.zeros(3)):
    if method == "oracle": T = alloc_oracle(mod, alpha_cmd, omega)
    elif method == "sym": T = alloc_sym(mod, alpha_cmd, omega)
    else: T = alloc_khop(mod, alpha_cmd, int(method[-1]), omega)
    return T


def allocate(mod, method, alpha_cmd, omega=np.zeros(3)):
    return np.clip(allocate_raw(mod, method, alpha_cmd, omega), 0, CF2X["Tmax"])


def alpha_max(mod, axis, Tm=None):
    """oracle（最小範數）不 clip 時、所有推力在 [0,Tmax] 內的最大 |α|（正負取小）。Tm=None 用 CF2X 的 Tmax。"""
    T0 = alloc_oracle(mod, np.zeros(3)); e = np.zeros(3); e[axis] = 1
    g = alloc_oracle(mod, e) - T0; Tm = CF2X["Tmax"] if Tm is None else Tm; best = np.inf
    for sgn in (1, -1):
        gg = sgn * g; pos = gg > 1e-15; neg = gg < -1e-15
        lim = min(np.min((Tm - T0[pos]) / gg[pos], initial=np.inf), np.min((0 - T0[neg]) / gg[neg], initial=np.inf))
        best = min(best, lim)
    return best


METHODS = ["oracle", "sym", "khop0", "khop1"]
MANEUVERS = [("hover", None, 0.0), ("pitch50", 1, 0.5), ("pitch100", 1, 1.0), ("roll100", 0, 1.0), ("yaw100", 2, 1.0)]


def evaluate(mod, method, axis, frac, amax, omega=np.zeros(3), kt=KT, kr=KR, jmethod="auto"):
    a_cmd = np.zeros(3)
    if axis is not None: a_cmd[axis] = frac * amax[axis]
    T = allocate(mod, method, a_cmd, omega)
    ext = module_loads(mod, T, omega)
    mx = max_over_joints(project(mod, joints(mod, ext, jmethod, kt, kr)))
    _, alpha = rigid_body(mod, T, omega)
    mx["achieved"] = alpha[axis] / a_cmd[axis] if axis is not None else np.nan
    raw = allocate_raw(mod, method, a_cmd, omega)
    mx["clipped"] = bool(np.any(raw < -1e-9) or np.any(raw > CF2X["Tmax"] + 1e-9))
    return mx


def run(kt=KT, kr=KR):
    """回傳 {(構型, 機動, 方法): 結果}、{構型: α_max}。payload 列只算 oracle，機動名 'payload'。"""
    res, amaxs = {}, {}
    for name, cells in CONFIGS.items():
        mod = build(cells); amax = [alpha_max(mod, a) for a in range(3)]; amaxs[name] = amax
        for mn, axis, frac in MANEUVERS:
            for meth in METHODS: res[(name, mn, meth)] = evaluate(mod, meth, axis, frac, amax, kt=kt, kr=kr)
        if name.startswith("line"):
            host = int(np.argmax([c[0] for c in cells]))
            pm = build(cells, payload=PAYLOAD_FRAC / (1 - PAYLOAD_FRAC) * CF2X["mass"] * len(cells), host=host)
            res[(name, "payload", "oracle")] = evaluate(pm, "oracle", None, 0.0, amax, kt=kt, kr=kr)
    return res, amaxs


def sensitivity():
    """閉迴路構型：kr 與 kt 比值 ×10、÷10 時，各格最大內力的變化。回傳 {構型: (Δ力/mg, Δ矩/(mg·pitch), 相對變化最大值)}。"""
    out = {}
    base, _ = run()
    for scale in (10.0, 0.1):
        alt, _ = run(KT, KR * scale)
        for key, b in base.items():
            if is_tree(build(CONFIGS[key[0]])): continue
            a = alt[key]; nm = key[0]
            f = max(abs(a[k] - b[k]) for k in KEYS[:3]) / (CF2X["mass"] * G)
            mo = max(abs(a[k] - b[k]) for k in KEYS[3:]) / (CF2X["mass"] * G * MODULE_PITCH)
            rel = max([abs(a[k] - b[k]) / b[k] for k in KEYS if b[k] > 1e-3 * (CF2X["mass"] * G if k in KEYS[:3] else CF2X["mass"] * G * MODULE_PITCH)], default=0.0)
            o = out.setdefault(nm, [0, 0, 0])
            out[nm] = [max(o[0], f), max(o[1], mo), max(o[2], rel)]
    return out


def fmt(x): return "%.2e" % x


if __name__ == "__main__":
    res, amaxs = run(); mg = CF2X["mass"] * G; mgp = mg * MODULE_PITCH
    L = ["| 構型 | 機動 | 分配 | 達成率 | 拉力 N | 面內剪力 N | 面外剪力 N | 彎矩 N·m | 扭矩 N·m |", "|---|---|---|---|---|---|---|---|---|"]
    for (n, mn, me), r in res.items():
        if n == "single": continue
        ach = "—" if np.isnan(r["achieved"]) else "%.2f%s" % (r["achieved"], "*" if r["clipped"] else "")
        L.append(f"| {n} | {mn} | {me} | {ach} | " + " | ".join(fmt(r[k]) for k in KEYS) + " |")
    Fk = lambda r: max(r[k] for k in KEYS[:3])
    # oracle 摘要：力 / mg，以及 / 「完全沒抵消時單模組要承受的慣性力」m·α·max|d|
    def base(mod, ax, amax_ax):
        dmax = np.abs(mod["d"][:, 1 if ax == 0 else 0]).max()      # pitch(α_y) 的 α×d_z = −α·dx；roll(α_x) 為 α·dy
        return mod["m"] * amax_ax * dmax
    S = ["| 構型 | N | 樹狀 | pitch100 力/mg | pitch100 力/未抵消慣性力 | roll100 力/mg | roll100 力/未抵消慣性力 | pitch100 彎矩或扭矩/(mg·pitch) | yaw100 面內剪力/mg |", "|---|---|---|---|---|---|---|---|---|"]
    rel_p, rel_r = [], []
    for n, c in CONFIGS.items():
        if n == "single": continue
        mod = build(c); r = res[(n, "pitch100", "oracle")]; rr = res[(n, "roll100", "oracle")]; y = res[(n, "yaw100", "oracle")]
        bp, br = base(mod, 1, amaxs[n][1]), base(mod, 0, amaxs[n][0])
        if bp > 1e-12: rel_p.append(Fk(r) / bp)
        if br > 1e-12: rel_r.append(Fk(rr) / br)
        S.append(f"| {n} | {mod['N']} | {'是' if is_tree(mod) else '否'} | {Fk(r)/mg:.3f} | " + (f"{Fk(r)/bp:.3f}" if bp > 1e-12 else "—") +
                 f" | {Fk(rr)/mg:.3f} | " + (f"{Fk(rr)/br:.3f}" if br > 1e-12 else "—") + f" | {max(r['bend'], r['torsion'])/mgp:.3f} | {y['shear_ip']/mg:.3f} |")
    # yaw 表
    Y = ["| 構型 | " + " | ".join(f"{m} 面內剪力/mg（達成率）" for m in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for n in CONFIGS:
        if n == "single": continue
        Y.append(f"| {n} | " + " | ".join(f"{res[(n,'yaw100',m)]['shear_ip']/mg:.3f}（{res[(n,'yaw100',m)]['achieved']:.2f}）" for m in METHODS) + " |")
    # 每格 α_max
    AM = ["| 構型 | roll_max | pitch_max | yaw_max (rad/s²) |", "|---|---|---|---|"] + \
         [f"| {n} | " + " | ".join("%.1f" % v for v in a) + " |" for n, a in amaxs.items()]
    # 靈敏度
    sens = sensitivity()
    SE = ["| 構型 | 最大內力變化 / mg | 最大內力矩變化 / (mg·pitch) | 相對變化最大值（僅統計 >0.1% 參考尺度的量） |", "|---|---|---|---|"] + \
         [f"| {n} | {v[0]:.3g} | {v[1]:.3g} | {v[2]:.3g} |" for n, v in sens.items()]
    # 主張判定用的數字
    nm = [n for n in CONFIGS if n != "single"]
    hov = max(Fk(res[(n, "hover", m)]) for n in nm for m in METHODS) / mg
    orc = {(n, mn): Fk(res[(n, mn, "oracle")]) / mg for n in nm for mn in ("pitch50", "pitch100", "roll100")}
    omax = max(orc.items(), key=lambda t: t[1])
    orc_m = max(max(res[(n, mn, "oracle")]["bend"], res[(n, mn, "oracle")]["torsion"]) for n in nm for mn in ("pitch100", "roll100")) / mgp
    sym_r = [Fk(res[(n, "pitch100", "sym")]) / mg for n in nm]
    kh = [Fk(res[(n, "pitch100", m)]) / res[(n, "pitch100", m)]["achieved"] / (Fk(res[(n, "pitch100", "oracle")])) for n in nm for m in ("khop0", "khop1") if Fk(res[(n, "pitch100", "oracle")]) > 1e-9]
    pay = {n: Fk(r) / mg for (n, mn, _), r in res.items() if mn == "payload"}
    yaw_all = all(res[(n, "yaw100", m)]["shear_ip"] > 1e-6 for n in nm for m in METHODS)
    yaw_unit = [res[(n, "yaw100", m)]["shear_ip"] / abs(res[(n, "yaw100", m)]["achieved"] * amaxs[n][2]) / CF2X["mass"] for n in nm for m in METHODS]
    claim = f"""**主張 A（oracle 內力 ≈ 0）：部分成立。**
- hover：所有構型、所有方法的接頭內力 ≤ {hov:.1e}·mg（= 0，每個模組自己扛自己的重量）。
- 最大角加速度（pitch/roll 50%、100%）下，oracle 所有接頭最大力 ≤ {omax[1]:.3f}·mg（{omax[0][0]} {omax[0][1]}），彎矩/扭矩 ≤ {orc_m:.3f}·mg·pitch。
  相對「完全沒抵消時單模組要承受的慣性力」m·α·max|d|，pitch100 殘差為 {min(rel_p):.3f}–{max(rel_p):.3f}、roll100 為 {min(rel_r):.3f}–{max(rel_r):.3f}（即抵消掉約 {100*(1-max(rel_p+rel_r)):.0f}% 以上）。
  所以「趨近 0」在「與 mg 或未抵消的慣性力比」都成立到個位數百分比，但嚴格說不是 0。
- 不為 0 的原因（解析可推）：模組 i 推力和的增量 ∝ r_i 成立；但抵消需要 Iyy_模組 = m·a²（模組內轉子分布的轉動慣量要等於把質量集中在轉子上），
  實際 Iyy = 1.4e-5 vs m·a² = 2.1e-5，所以差一個固定比例，殘差 ∝ α·|r_i|（line3 手算驗證見測試）。
- 內力主要來自分配和幾何不一致：成立。pitch100 下 sym（每模組相同推力）最大力 {min(sym_r):.2f}–{max(sym_r):.2f}·mg（且有 clip、達成率不到 1），
  k 跳規則在「每單位達成角加速度」下，內力是 oracle 的 {min(kh):.1f}–{max(kh):.1f} 倍（k 越大越接近 oracle，ball=全架時相同）。
- 但 oracle 並非總是 ≈ 0：有 payload 時（oracle 已知新 CoM 與 J），line2/3/4/6/8 的 hover 最大力為 {pay['line2']:.2f}/{pay['line3']:.2f}/{pay['line4']:.2f}/{pay['line6']:.2f}/{pay['line8']:.2f}·mg。
  原因：最小範數解不看質量分布，推力與質量不成比例（均勻質量的前提被打破），重量差要由接頭傳遞，且隨線長以槓桿累積。

**主張 B（yaw 剪力每個方法都有）：成立，但機制和主張敘述不同。**
- yaw100 下所有構型（single 除外）、所有方法的面內剪力皆 > 0（{'是' if yaw_all else '否'}，見第 3 節）。
- 機制：面內慣性力 −m(α×d_i) 方向在水平面，而推力沿 z、無法抵消，所以面內剪力與分配無關，只取決於實得 α_z（k 跳規則達成率低則剪力成比例變小）。
  最大面內剪力 = {min(yaw_unit):.2f}–{max(yaw_unit):.2f} m × (m_模組·α_z)（子樹的 Σ|d| 累加，長度單位）。
- yaw 反扭矩「每模組相同、需求 ∝ r」造成的是繞 z 的力矩不平衡（∝ m|d|²α_z），體現在繞 z 的彎矩（面內彎矩）上，而不是剪力。"""
    md = f"""# P0d 接頭內力（解析）

模型見 `analysis/joint_loads.py` 檔頭 docstring。單位 N、N·m；模組間距 {MODULE_PITCH} m；
**參考尺度：單一模組 hover 重量 m·g = {mg:.4f} N**；力矩參考 m·g·pitch = {mgp:.2e} N·m。
表中「達成率」= 實得角加速度 / 指令；`*` 代表有推力被 clip。接頭勁度 kt={KT:g} N/m、kr={KR:g} N·m/rad（只影響含迴路構型）。
payload：質點，質量 = {PAYLOAD_FRAC:.0%} 含 payload 總質量，掛在 line 的 x 最大端模組中心，oracle 用新 CoM／J 分配。

## 1. 主張判定

{claim}

## 2. oracle 摘要

{chr(10).join(S)}

## 3. yaw100：各分配的面內剪力（/mg）與達成率

{chr(10).join(Y)}

## 4. 閉迴路構型的勁度靈敏度（kr ×10 與 ÷10，取兩者最大）

{chr(10).join(SE)}

## 5. 最大角加速度（oracle、不 clip）

{chr(10).join(AM)}

## 6. 全表（每個構型×機動×分配，所有接頭中的最大值）

{chr(10).join(L)}
"""
    open("results/p0/joint_loads.md", "w").write(md)
    print("\n".join(S)); print("\n".join(Y)); print("\n".join(SE))
    print(claim)
