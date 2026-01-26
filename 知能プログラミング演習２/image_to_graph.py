import cv2
import numpy as np
import networkx as nx
import pickle
from skimage.morphology import skeletonize

IMG_PATH = "node_env.png"

# -------------------------
# 1) 青ノード検出（まず拾う）
# -------------------------
def detect_blue_nodes(img, area_min=20, area_max=6000):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 60, 60])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nodes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area_min <= area <= area_max:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            nodes.append((cx,cy))
    return nodes

# -------------------------
# 2) 赤線マスク
# -------------------------
def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

# -------------------------
# 3) ノードを「赤線の近く」に限定（誤検出除去）
# -------------------------
def filter_nodes_near_red(nodes, redmask, max_dist=9):
    inv = (redmask == 0).astype(np.uint8)  # 赤=0, その他=1
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    keep = []
    for (x,y) in nodes:
        if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1]:
            if dist[y,x] <= max_dist:
                keep.append((x,y))
    return keep

# -------------------------
# 4) 近いノードをマージ（重複削減）
# -------------------------
def merge_close_nodes(nodes, r=8):
    merged = []
    for (x,y) in nodes:
        ok = True
        for (mx,my) in merged:
            if (x-mx)**2 + (y-my)**2 <= r*r:
                ok = False
                break
        if ok:
            merged.append((x,y))
    return merged

# -------------------------
# 5) 赤線スケルトン化
# -------------------------
def skeletonize_red(redmask):
    skel = skeletonize(redmask > 0)  # Trueが線
    return skel.astype(np.uint8)     # 0/1

# -------------------------
# 6) ノードをスケルトン上にスナップ
# -------------------------
def snap_to_skeleton(nodes, skel, max_snap=10):
    ys, xs = np.where(skel > 0)
    skel_pts = np.stack([xs, ys], axis=1)  # (N,2) x,y

    snapped = []
    for (x,y) in nodes:
        d2 = np.sum((skel_pts - np.array([x,y]))**2, axis=1)
        k = int(np.argmin(d2))
        sx, sy = skel_pts[k]
        if d2[k] <= max_snap*max_snap:
            snapped.append((int(sx), int(sy)))
    return snapped

# -------------------------
# 7) スケルトン上で道沿い接続（BFS）
# -------------------------
def bfs_shortest_path_len(skel, p1, p2, max_steps=20000):
    h, w = skel.shape
    (x1,y1) = p1
    (x2,y2) = p2
    if not (0<=x1<w and 0<=y1<h and 0<=x2<w and 0<=y2<h):
        return None
    if skel[y1,x1] == 0 or skel[y2,x2] == 0:
        return None

    from collections import deque
    q = deque()
    q.append((x1,y1))
    dist = { (x1,y1): 0 }

    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while q:
        x,y = q.popleft()
        d = dist[(x,y)]
        if (x,y) == (x2,y2):
            return d
        if d >= max_steps:
            continue
        for dx,dy in dirs:
            nx_, ny_ = x+dx, y+dy
            if 0<=nx_<w and 0<=ny_<h and skel[ny_,nx_] > 0:
                if (nx_,ny_) not in dist:
                    dist[(nx_,ny_)] = d+1
                    q.append((nx_,ny_))
    return None

def build_graph_from_skeleton(nodes, skel, k_nearest=6):
    G = nx.Graph()
    pts = np.array(nodes, dtype=np.float32)

    for i,(x,y) in enumerate(nodes):
        G.add_node(i, pos=(x,y))

    for i in range(len(nodes)):
        d2 = np.sum((pts - pts[i])**2, axis=1)
        nn = np.argsort(d2)[1:k_nearest+1]
        for j in nn:
            if j <= i:
                continue

            path_len = bfs_shortest_path_len(skel, nodes[i], nodes[j], max_steps=20000)
            if path_len is None:
                continue

            G.add_edge(i, j, distance=float(path_len), risk=1.0, blocked=False)

    return G

# -------------------------
# 追加A) 赤線が線分上にどれだけあるか（ブリッジ用）
# -------------------------
def line_red_ratio(redmask, p1, p2, samples=220, thickness=3):
    x1,y1 = p1; x2,y2 = p2
    hit = 0
    h, w = redmask.shape

    for i in range(samples):
        t = i/(samples-1)
        x = int(x1*(1-t) + x2*t)
        y = int(y1*(1-t) + y2*t)

        x0 = max(0, x-thickness); x1b = min(w-1, x+thickness)
        y0 = max(0, y-thickness); y1b = min(h-1, y+thickness)

        if (redmask[y0:y1b+1, x0:x1b+1] > 0).any():
            hit += 1

    return hit / samples

# -------------------------
# 追加B) 連結成分を赤線で橋渡し（ここが「繋げたい部分をつなぐ」）
# -------------------------
def bridge_components_by_red(G, nodes_xy, redmask,
                            k_candidates=10,
                            ratio_th=0.22,
                            max_new_edges=120,
                            max_dist=500):
    """
    大きい成分に小さい成分を吸収するように橋渡しエッジを追加。
    「赤線があるところだけ」繋ぐので暴走しにくい。
    """
    added = 0
    pts = np.array(nodes_xy, dtype=np.float32)

    while True:
        comps = list(nx.connected_components(G))
        if len(comps) <= 1:
            break

        comps.sort(key=len, reverse=True)
        base = np.array(list(comps[0]), dtype=int)
        others = [list(c) for c in comps[1:]]

        best = None  # (score, u, v, ratio, dist)

        for comp in others:
            comp = np.array(comp, dtype=int)

            # 近いペアだけ見るため、comp側の各vからbaseの近傍を探す
            for v in comp:
                d2 = np.sum((pts[base] - pts[v])**2, axis=1)
                nn = np.argsort(d2)[:k_candidates]
                for idx in nn:
                    u = int(base[idx])
                    dist = float(np.sqrt(d2[idx]))
                    if dist > max_dist:
                        continue

                    r = line_red_ratio(redmask, nodes_xy[u], nodes_xy[v])
                    if r < ratio_th:
                        continue

                    score = r - 0.0008 * dist
                    if best is None or score > best[0]:
                        best = (score, u, int(v), r, dist)

        if best is None:
            # 赤線条件で繋がる候補がない → ここで終了
            break

        _, u, v, r, dist = best
        if not G.has_edge(u, v):
            G.add_edge(u, v, distance=dist, risk=1.0, blocked=False)
            added += 1

        if added >= max_new_edges:
            break

    return added

# -------------------------
# 9) 可視化
# -------------------------
def visualize(img, nodes, G, out="nit_graph_debug.png"):
    vis = img.copy()

    for i,(x,y) in enumerate(nodes):
        cv2.circle(vis,(x,y),4,(0,255,0),-1)
        cv2.putText(vis,str(i),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),2)

    for u,v in G.edges():
        x1,y1 = nodes[u]
        x2,y2 = nodes[v]
        cv2.line(vis,(x1,y1),(x2,y2),(255,0,0),2)

    cv2.imwrite(out, vis)
    print("Saved", out)

# -------------------------
# main
# -------------------------
if __name__=="__main__":
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {IMG_PATH}")

    red = red_mask(img)
    raw_nodes = detect_blue_nodes(img)

    # 誤検出除去 → マージ → スケルトンへスナップ（※ここはあなたの原案維持）
    nodes = filter_nodes_near_red(raw_nodes, red, max_dist=9)
    nodes = merge_close_nodes(nodes, r=8)

    skel = skeletonize_red(red)
    nodes = snap_to_skeleton(nodes, skel, max_snap=10)

    print("Detected nodes (filtered/snapped):", len(nodes))

    # まずスケルトンBFSで作る（確実な部分）
    G = build_graph_from_skeleton(nodes, skel, k_nearest=6)

    print("Before bridging:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges",
          "components:", nx.number_connected_components(G))

    # ★ここが完成形の肝：分断を赤線で橋渡し
    added = bridge_components_by_red(G, nodes, red,
                                    k_candidates=10,
                                    ratio_th=0.22,
                                    max_new_edges=120,
                                    max_dist=520)
    print("bridging edges added:", added)

    print("After bridging:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges",
          "components:", nx.number_connected_components(G))

    comps = list(nx.connected_components(G))
    sizes = sorted([len(c) for c in comps], reverse=True)
    print("component sizes (top10):", sizes[:10])

    # 必要なら最大成分だけにする（RLを安定させたい場合はTrue）
    USE_LARGEST = True
    if USE_LARGEST and nx.number_connected_components(G) > 1:
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        print("Kept largest component:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

    visualize(img, nodes, G, out="nit_graph_debug_fixed.png")

    # 保存
    with open("nit_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    with open("nit_nodes.pkl", "wb") as f:
        pickle.dump(nodes, f)

    print("Saved nit_graph.pkl and nit_nodes.pkl")
