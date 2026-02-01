import pickle
import random
import cv2
import numpy as np
import networkx as nx

from env_graph import GraphEvacuationEnv
from q_learning import QLearningAgent

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

IMG_PATH = "node_env.png"

# 1) 災害ノードを「重み付きランダム」で1つ選ぶ
#   - 例：次数(degree)が大きいノードほど危険（人が集まりやすい想定）
def pick_disaster_node(G, avoid_nodes=None, power=1.6, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    avoid_nodes = set(avoid_nodes or [])
    candidates = [n for n in G.nodes() if n not in avoid_nodes]
    if not candidates:
        raise RuntimeError("災害ノード候補が空です（avoid_nodes が多すぎる可能性）。")

    deg = np.array([G.degree(n) for n in candidates], dtype=np.float64)
    # degree^power（中心っぽいほど重みUP）
    w = np.power(deg + 1.0, power)

    # 全部同じになった場合の保険
    if w.sum() <= 0:
        w = np.ones_like(w)

    p = w / w.sum()
    disaster = int(np.random.choice(candidates, size=1, replace=False, p=p)[0])
    return disaster

def nearest_node_from_pixel(nodes_xy, point_xy, candidate_nodes=None):
    """画像のピクセル座標 (x, y) に最も近いノードIDを返す"""
    if point_xy is None:
        return None
    px, py = point_xy
    if candidate_nodes is None:
        candidate_nodes = range(len(nodes_xy))
    best = None  # (d2, node)
    for n in candidate_nodes:
        x, y = nodes_xy[n]
        d2 = (x - px) ** 2 + (y - py) ** 2
        if best is None or d2 < best[0]:
            best = (d2, n)
    return int(best[1]) if best else None

# 2) 災害ノード周辺に START（逃げる人）を置く
#   - ピクセル距離で「近いノード」を候補にする
def pick_start_near_disaster(G, nodes_xy, disaster, max_px=180, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dx, dy = nodes_xy[disaster]
    near = []
    for n in G.nodes():
        if n == disaster:
            continue
        x, y = nodes_xy[n]
        d = ((x - dx)**2 + (y - dy)**2) ** 0.5
        if d <= max_px:
            near.append(n)

    # 近場が無ければ全体から（保険）
    if not near:
        near = [n for n in G.nodes() if n != disaster]

    return int(random.choice(near))

# 3) STARTから到達可能で、なるべく遠い GOAL を選ぶ
def pick_goal_far_reachable(G, start, tries=80, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes = list(G.nodes())
    best = None  # (dist, goal)

    for _ in range(tries):
        g = random.choice(nodes)
        if g == start:
            continue
        try:
            d = nx.shortest_path_length(G, start, g, weight="distance")
        except nx.NetworkXNoPath:
            continue
        if best is None or d > best[0]:
            best = (d, g)

    if best is None:
        # 保険：到達可能なものを適当に
        reach = nx.single_source_shortest_path_length(G, start)
        reach_nodes = [n for n in reach.keys() if n != start]
        if not reach_nodes:
            raise RuntimeError("STARTから到達可能なGOALが見つかりません。")
        return int(random.choice(reach_nodes))

    return int(best[1])

# 4) 災害をグラフに反映（災害に近いエッジほど risk を増やす）
#   - env_graph.py は edge の risk を参照して報酬に反映する :contentReference[oaicite:0]{index=0}
def apply_disaster_risk(G, nodes_xy, disaster, sigma=140.0, risk_scale=6.0):
    dx, dy = nodes_xy[disaster]

    for u, v, data in G.edges(data=True):
        x1, y1 = nodes_xy[u]
        x2, y2 = nodes_xy[v]

        # エッジの端点のうち近い方の距離で近さを測る（軽量）
        du = ((x1 - dx)**2 + (y1 - dy)**2) ** 0.5
        dv = ((x2 - dx)**2 + (y2 - dy)**2) ** 0.5
        d = min(du, dv)

        # 近いほど追加リスクが大きい（0〜risk_scale）
        add = risk_scale * np.exp(-d / sigma)

        base = float(data.get("risk", 1.0))
        data["risk"] = base + float(add)

    # ノード自体も「ここが災害」として見せたいので返すだけ
    return G

def remove_nodes_near_disaster(G, nodes_xy, disaster, radius_px=120, keep_nodes=None):
    """
    災害ノードの周囲radius_px以内のノードをグラフから除外する。
    keep_nodes に入っているノードは除外しない。
    """
    keep_nodes = set(keep_nodes or [])
    dx, dy = nodes_xy[disaster]
    to_remove = []
    for n in list(G.nodes()):
        if n in keep_nodes:
            continue
        x, y = nodes_xy[n]
        d = ((x - dx) ** 2 + (y - dy) ** 2) ** 0.5
        if d <= radius_px:
            to_remove.append(n)
    if to_remove:
        G.remove_nodes_from(to_remove)
    return len(to_remove)


def greedy_path_from_Q(env, agent, max_steps=500):
    s = env.reset()
    path = [s]
    for _ in range(max_steps):
        acts = env.actions(s)
        if not acts:
            break
        a = max(acts, key=lambda x: agent.Q[(s, x)])
        s, _, done, _ = env.step(a)
        path.append(s)
        if done:
            break
    return path

def draw_path_on_image(img_path, nodes_xy, path, out_path="evac_path.png", disaster=None, label_nodes=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {img_path}")

    # 経路を緑で描画
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        x1, y1 = nodes_xy[u]
        x2, y2 = nodes_xy[v]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # START/GOALを強調
    sx, sy = nodes_xy[path[0]]
    gx, gy = nodes_xy[path[-1]]
    cv2.circle(img, (sx, sy), 10, (0, 255, 255), -1)  # start: yellow
    cv2.circle(img, (gx, gy), 10, (255, 255, 0), -1)  # goal: cyan

    if disaster is not None:
        dx, dy = nodes_xy[disaster]
        cv2.circle(img, (dx, dy), 12, (0, 0, 255), -1)  # disaster: red

    # ノードIDを表示（必要なときだけ）
    if label_nodes:
        # 全ノードに表示したい場合は nodes_xy を使う
        for n, (x, y) in enumerate(nodes_xy):
            if x is None or y is None:
                continue
            x, y = nodes_xy[n]
            cv2.putText(
                img,
                str(n),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                str(n),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(out_path, img)
    print("Saved", out_path)

# 7) ローカルLLMで避難指示文を生成
def build_llm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
    return device, tokenizer, model

def generate_evacuation_text(device, tokenizer, model, nodes_xy, path, disaster, start, goal):
    # 座標列（見やすく短く）
    coords = [nodes_xy[n] for n in path]
    # 長すぎるとモデルが崩れるので間引く
    if len(coords) > 30:
        coords = coords[:: max(1, len(coords)//30)]

    sx, sy = nodes_xy[start]
    gx, gy = nodes_xy[goal]
    dx, dy = nodes_xy[disaster]

    prompt = f"""
あなたは防災ナビゲーションAIです。
地図上のノード座標(ピクセル)に基づいて、避難指示を日本語で作ってください。

条件:
- 箇条書き
- 5〜9行
- 最初に「災害地点から離れる」ことを明記
- 座標は必要最小限

災害地点: ({dx},{dy})
開始地点: ({sx},{sy})
避難所: ({gx},{gy})
経路(一部): {coords}

避難指示:
"""

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
    out = model.generate(input_ids, max_length=220, do_sample=True, top_p=0.92, temperature=0.9)
    text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    if not text:
        # 保険（LLMが空出力のとき）
        text = "災害地点から離れる方向に移動し、経路に沿って避難場所へ向かってください。"
    return text

def build_simple_instructions(nodes_xy, path, max_steps=20):
    """
    ノードIDと曲がり方向で簡易な避難指示を作る。
    - 直進/右折/左折/Uターンを判定
    - 方角（東西南北）も補助的に出す
    """
    if not path or len(path) < 2:
        return ["経路が短すぎて指示を生成できません。"]

    def dir_from_delta(dx, dy):
        if abs(dx) >= abs(dy):
            return "東" if dx > 0 else "西"
        return "南" if dy > 0 else "北"

    def heading_from_vec(v):
        dx, dy = v
        if abs(dx) >= abs(dy):
            return "東" if dx > 0 else "西"
        return "南" if dy > 0 else "北"

    def turn_from_headings(h1, h2):
        order = ["北", "東", "南", "西"]
        i1 = order.index(h1)
        i2 = order.index(h2)
        diff = (i2 - i1) % 4
        if diff == 0:
            return "直進"
        if diff == 2:
            return "Uターン"
        return "右折" if diff == 1 else "左折"

    instr = [f"開始: ノード {path[0]}"]

    # 方向が同じ区間はまとめる
    prev_vec = None
    prev_heading = None
    run_start = path[0]
    run_dir = None
    run_turn = None

    def flush_run(end_node):
        if run_dir is None:
            return
        if run_turn is None:
            instr.append(f"{run_start} → {end_node}: {run_dir}へ進む")
        else:
            instr.append(f"{run_start} で{run_turn}し、{run_dir}へ直進して {end_node} まで")

    limit = min(len(path), max_steps + 1)
    for i in range(1, limit):
        u = path[i - 1]
        v = path[i]
        x1, y1 = nodes_xy[u]
        x2, y2 = nodes_xy[v]
        vec = (x2 - x1, y2 - y1)
        direction = heading_from_vec(vec)

        if run_dir is None:
            run_start = u
            run_dir = direction
            run_turn = None if prev_heading is None else turn_from_headings(prev_heading, direction)
        elif direction != run_dir:
            flush_run(u)
            run_start = u
            run_dir = direction
            run_turn = None if prev_heading is None else turn_from_headings(prev_heading, direction)

        prev_vec = vec
        prev_heading = direction

    # 最後の区間を出力
    flush_run(path[limit - 1])

    if len(path) > max_steps + 1:
        instr.append("...（経路が長いため一部省略）")
    instr.append(f"到着: ノード {path[-1]}")
    return instr

def choose_nearer_shelter(G, start_id, shelter_ids, weight="length"):
    """
    start_id から shelter_ids の各避難所までの最短経路長を計算し、
    もっとも近い避難所ノードIDと距離リストを返す。

    weight:
      - OSMnxの徒歩グラフなら通常 "length"（メートル）
      - あなたのグラフで "distance" を使っているなら "distance"
    """
    lengths = []
    for sid in shelter_ids:
        try:
            L = nx.shortest_path_length(G, start_id, sid, weight=weight)
        except nx.NetworkXNoPath:
            L = float("inf")
        lengths.append(L)

    best = int(shelter_ids[int(np.argmin(lengths))])
    return best, lengths

def main():
    with open("nit_graph.pkl", "rb") as f:
        G = pickle.load(f)
    with open("nit_nodes.pkl", "rb") as f:
        nodes_xy = pickle.load(f)  # list of (x,y) with index=node_id

    # 手動でつなぎたいエッジを追加
    if not G.has_edge(14, 6):
        x1, y1 = nodes_xy[14]
        x2, y2 = nodes_xy[6]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(14, 6, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(25, 26):
        x1, y1 = nodes_xy[25]
        x2, y2 = nodes_xy[26]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(25, 26, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(38, 55):
        x1, y1 = nodes_xy[38]
        x2, y2 = nodes_xy[55]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(38, 55, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(73, 97):
        x1, y1 = nodes_xy[73]
        x2, y2 = nodes_xy[97]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(73, 97, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(79, 82):
        x1, y1 = nodes_xy[79]
        x2, y2 = nodes_xy[82]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(79, 82, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(80, 81):
        x1, y1 = nodes_xy[80]
        x2, y2 = nodes_xy[81]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(80, 81, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(30, 32):
        x1, y1 = nodes_xy[30]
        x2, y2 = nodes_xy[32]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(30, 32, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(36, 38):
        x1, y1 = nodes_xy[36]
        x2, y2 = nodes_xy[38]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(36, 38, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(7, 16):
        x1, y1 = nodes_xy[7]
        x2, y2 = nodes_xy[16]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(7, 16, distance=dist, risk=1.0, blocked=False)
    if not G.has_edge(26, 28):
        x1, y1 = nodes_xy[26]
        x2, y2 = nodes_xy[28]
        dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        G.add_edge(26, 28, distance=dist, risk=1.0, blocked=False)

    print("Graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

    comps = list(nx.connected_components(G))
    largest = max(comps, key=len)
    Gs = G.subgraph(largest).copy()

    # (A) 災害ノードを選ぶ（重み付きランダム）
    disaster = pick_disaster_node(Gs, avoid_nodes=None, seed=None)

    # (B) STARTを災害の近くに置く（逃げる人が災害付近にいる想定）
    START = pick_start_near_disaster(Gs, nodes_xy, disaster, max_px=180)

    # 避難所（ノードIDで指定）
    shelter_ground = 107  # グラウンド
    shelter_b2 = 26       # 2号館前

    shelters = [shelter_ground, shelter_b2]

    # 災害が避難所のどちらかで発生した場合は、反対側へ避難
    if disaster == shelter_ground:
        GOAL = shelter_b2
        dists = None
    elif disaster == shelter_b2:
        GOAL = shelter_ground
        dists = None
    else:
        best_shelter, dists = choose_nearer_shelter(Gs, START, shelters, weight="distance")
        GOAL = best_shelter

    print("Shelter nodes:", shelters)
    print("Distances to shelters:", dists)
    print("Chosen shelter:", GOAL)

    # 災害ノードを通らないように、災害ノードをグラフから除外
    # （これで経路上に災害ノードが含まれなくなる）
    Gs_safe = Gs.copy()
    if disaster in Gs_safe.nodes():
        Gs_safe.remove_node(disaster)

    # 災害ノード周辺を通行禁止にする半径（ピクセル）
    NO_GO_RADIUS_PX = 120
    removed = remove_nodes_near_disaster(
        Gs_safe,
        nodes_xy,
        disaster,
        radius_px=NO_GO_RADIUS_PX,
        keep_nodes={START, shelter_ground, shelter_b2},
    )
    print("Removed nodes near disaster:", removed)

    if START not in Gs_safe.nodes():
        raise RuntimeError("災害ノード除外により START が到達不能になりました。")

    # もし近い避難所が到達不可なら、もう一方に切り替える
    if GOAL not in Gs_safe.nodes() or not nx.has_path(Gs_safe, START, GOAL):
        alt = shelter_b2 if GOAL == shelter_ground else shelter_ground
        if alt in Gs_safe.nodes() and nx.has_path(Gs_safe, START, alt):
            GOAL = alt
        else:
            # どちらも不可なら START を選び直す
            retry = 0
            max_retry = 50
            while retry < max_retry:
                START = pick_start_near_disaster(Gs_safe, nodes_xy, disaster, max_px=180)
                if START in Gs_safe.nodes():
                    if (shelter_ground in Gs_safe.nodes() and nx.has_path(Gs_safe, START, shelter_ground)) or \
                       (shelter_b2 in Gs_safe.nodes() and nx.has_path(Gs_safe, START, shelter_b2)):
                        if shelter_ground in Gs_safe.nodes() and nx.has_path(Gs_safe, START, shelter_ground):
                            GOAL = shelter_ground
                        else:
                            GOAL = shelter_b2
                        break
                retry += 1
            if retry >= max_retry:
                raise RuntimeError("災害ノード除外により避難場所へ到達できません（START再抽選も失敗）。")

    
    print("DISASTER:", disaster, "pos:", nodes_xy[disaster])
    print("START:", START, "pos:", nodes_xy[START])
    print("GOAL:", GOAL, "pos:", nodes_xy[GOAL])
    print("has_path:", nx.has_path(Gs_safe, START, GOAL))

    # すでに避難所にいる場合は移動不要
    if START == GOAL:
        print("\n===== 避難指示（ノードID/曲がり） =====")
        print(f"災害ノード: {disaster}")
        print(f"- 開始/到着: ノード {START}（すでに避難場所）")
        # 描画（経路なし）
        draw_path_on_image(IMG_PATH, nodes_xy, [START], out_path="evac_rl.png", disaster=disaster, label_nodes=True)
        return

    # 災害をriskとしてグラフに反映
    apply_disaster_risk(Gs_safe, nodes_xy, disaster, sigma=140.0, risk_scale=6.0)

    # まず最短路（比較用：RLの妥当性チェック）
    sp = nx.shortest_path(Gs_safe, START, GOAL, weight="distance")
    sp_cost = nx.path_weight(Gs_safe, sp, weight="distance")
    print("ShortestPath cost:", sp_cost, "len:", len(sp))

    # RL環境 & 学習
    env = GraphEvacuationEnv(
        Gs_safe, start=START, goal=GOAL,
        risk_weight=15.0,
        step_penalty=1.0,     # 曲がり/歩数より距離を優先
        goal_reward=8000.0,
        revisit_penalty=5.0,  # ループ抑制は弱めに
        backtrack_penalty=150.0,
        shaping_scale=8.0     # 距離短縮をより重視
    )
    agent = QLearningAgent(alpha=0.2, gamma=0.97,
                           epsilon=1.0, epsilon_min=0.10, epsilon_decay=0.9997)

    episodes = 20000
    max_steps = 700

    for ep in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            a = agent.choose_action(env, s)
            if a is None:
                break
            s2, r, done, _ = env.step(a)
            agent.update(env, s, a, r, s2, done)
            s = s2
            if done:
                break
        agent.decay_epsilon()

        # 途中経過を少し表示（重いなら消してOK）
        if (ep + 1) % 1000 == 0:
            path = greedy_path_from_Q(env, agent, max_steps=800)
            ok = (path[-1] == GOAL)
            print(f"ep {ep+1}: greedy_reach_goal={ok}, path_len={len(path)}, eps={agent.epsilon:.3f}")

    # 学習結果（greedy）
    rl_path = greedy_path_from_Q(env, agent, max_steps=800)
    if rl_path[-1] != GOAL:
        print("[WARN] RLがGOALに到達できていません。グラフ分断 or 学習不足の可能性。")
    else:
        rl_cost = nx.path_weight(Gs_safe, rl_path, weight="distance")
        print("RL path cost:", rl_cost, "len:", len(rl_path))

    print("RL path:", rl_path)

    # 描画：最短路とRL路を比較したければ2枚作る
    draw_path_on_image(IMG_PATH, nodes_xy, sp, out_path="evac_shortest.png", disaster=disaster, label_nodes=False)
    draw_path_on_image(IMG_PATH, nodes_xy, rl_path, out_path="evac_rl.png", disaster=disaster, label_nodes=True)

    # 分かりやすい避難指示（ノードID + 曲がり）
    print("\n===== 避難指示（ノードID/曲がり） =====")
    print(f"災害発生場所(ノードID): {disaster}")
    for line in build_simple_instructions(nodes_xy, rl_path, max_steps=25):
        print("-", line)

    # LLMで避難指示（必要なら有効化）
    # device, tokenizer, model = build_llm()
    # evac_text = generate_evacuation_text(device, tokenizer, model, nodes_xy, rl_path, disaster, START, GOAL)
    # print("\n===== ローカルLLMによる避難指示 =====")
    # print(evac_text)

if __name__ == "__main__":
    main()
