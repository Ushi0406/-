import pickle
import random
import cv2
import numpy as np
import networkx as nx

from env_graph import GraphEvacuationEnv
from q_learning import QLearningAgent

IMG_PATH = "node_env.png"

def pick_start_goal_same_component(G, prefer_far=True):
    # 最大連結成分だけに絞る（到達不能を防ぐ）
    comps = list(nx.connected_components(G))
    largest = max(comps, key=len)
    nodes = list(largest)

    if len(nodes) < 2:
        raise RuntimeError("連結成分のノード数が少なすぎます。")

    if not prefer_far:
        s, g = random.sample(nodes, 2)
        return s, g, largest

    # なるべく遠いペアを探す（デモで経路が分かりやすい）
    # ランダムに何回か試して最長を採用
    best = None
    for _ in range(60):
        s, g = random.sample(nodes, 2)
        try:
            d = nx.shortest_path_length(G, s, g, weight="distance")
        except nx.NetworkXNoPath:
            continue
        if best is None or d > best[0]:
            best = (d, s, g)
    if best is None:
        s, g = random.sample(nodes, 2)
        return s, g, largest
    _, s, g = best
    return s, g, largest

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

def draw_path_on_image(img_path, nodes_xy, path, out_path="evac_path.png"):
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

    cv2.imwrite(out_path, img)
    print("Saved", out_path)

def main():
    with open("nit_graph.pkl", "rb") as f:
        G = pickle.load(f)
    with open("nit_nodes.pkl", "rb") as f:
        nodes_xy = pickle.load(f)  # list of (x,y) with index=node_id

    print("Graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

    # START/GOALを「必ず到達可能」になるように選ぶ
    START, GOAL, largest = pick_start_goal_same_component(G, prefer_far=True)
    print("START:", START, "GOAL:", GOAL, "component size:", len(largest))
    print("has_path:", nx.has_path(G, START, GOAL))

    # まず最短路（比較用：RLの妥当性チェック）
    sp = nx.shortest_path(G, START, GOAL, weight="distance")
    sp_cost = nx.path_weight(G, sp, weight="distance")
    print("ShortestPath cost:", sp_cost, "len:", len(sp))

    # RL環境 & 学習
    env = GraphEvacuationEnv(
        G, start=START, goal=GOAL,
        risk_weight=15.0,
        step_penalty=1.0,
        goal_reward=8000.0,
        revisit_penalty=10.0,
        backtrack_penalty=20.0,
        shaping_scale=5.0
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
        rl_cost = nx.path_weight(G, rl_path, weight="distance")
        print("RL path cost:", rl_cost, "len:", len(rl_path))

    print("RL path:", rl_path)

    # 描画：最短路とRL路を比較したければ2枚作る
    draw_path_on_image(IMG_PATH, nodes_xy, sp, out_path="evac_shortest.png")
    draw_path_on_image(IMG_PATH, nodes_xy, rl_path, out_path="evac_rl.png")

if __name__ == "__main__":
    main()
