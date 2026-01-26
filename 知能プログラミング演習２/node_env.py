import osmnx as ox
import matplotlib.pyplot as plt
import contextily as ctx

place_name = "名古屋工業大学, 愛知県, 日本"

# ① OSM から徒歩ネットワーク取得
G = ox.graph_from_place(place_name, network_type="walk")

# ② GeoDataFrame に変換
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

# ③ Web Mercator（必須）
gdf_edges = gdf_edges.to_crs(epsg=3857)
gdf_nodes = gdf_nodes.to_crs(epsg=3857)
# --- プロット量削減設定 ---
EDGE_LENGTH_QUANTILE = 0.25  # 下位何割の短いエッジを削除 (0-1)
EDGE_SAMPLE_FRAC = 0.6       # 残ったエッジをランダムサンプリング (0-1)
SIMPLIFY_TOLERANCE = 5       # メートル単位で線を簡略化（0で無効化）
NODE_SAMPLE_FRAC = 0.5       # u/v が無い場合にノードをランダムサンプリング
# ----------------------------

# 出力ファイル名
nodes_csv = "nodes.csv"
edges_csv = "edges.csv"
nodes_txt = "nodes.txt"
edges_txt = "edges.txt"

# フルデータを CSV / テキストで保存（後で reduced も保存）
gdf_nodes.to_csv(nodes_csv, index=True)
gdf_edges.to_csv(edges_csv, index=True)

with open(nodes_txt, "w", encoding="utf-8") as f:
    f.write(gdf_nodes.to_string())

with open(edges_txt, "w", encoding="utf-8") as f:
    f.write(gdf_edges.to_string())

print(f"Saved nodes to {nodes_csv} and {nodes_txt}")
print(f"Saved edges to {edges_csv} and {edges_txt}")

# --- エッジ長を計算して短辺を除外 ---
gdf_edges = gdf_edges.copy()
gdf_edges['length_m'] = gdf_edges.geometry.length
length_thr = gdf_edges['length_m'].quantile(EDGE_LENGTH_QUANTILE)
gdf_edges_reduced = gdf_edges[gdf_edges['length_m'] >= length_thr]

# サンプリング
if 0 < EDGE_SAMPLE_FRAC < 1 and len(gdf_edges_reduced) > 0:
    gdf_edges_reduced = gdf_edges_reduced.sample(frac=EDGE_SAMPLE_FRAC, random_state=1)

# ジオメトリ簡略化
if SIMPLIFY_TOLERANCE > 0 and len(gdf_edges_reduced) > 0:
    gdf_edges_reduced = gdf_edges_reduced.copy()
    gdf_edges_reduced.geometry = gdf_edges_reduced.geometry.simplify(SIMPLIFY_TOLERANCE)

# 残ったエッジに関係するノードのみ抽出（u/v カラムがある場合）
if 'u' in gdf_edges_reduced.columns and 'v' in gdf_edges_reduced.columns and len(gdf_edges_reduced) > 0:
    node_ids = set(gdf_edges_reduced['u'].tolist()) | set(gdf_edges_reduced['v'].tolist())
    gdf_nodes_reduced = gdf_nodes[gdf_nodes.index.isin(node_ids)]
else:
    # ノードが多い場合はサンプリング
    if 0 < NODE_SAMPLE_FRAC < 1:
        gdf_nodes_reduced = gdf_nodes.sample(frac=NODE_SAMPLE_FRAC, random_state=1)
    else:
        gdf_nodes_reduced = gdf_nodes.copy()

# 出力（プロット用に reduced CSV と TXT を保存）
gdf_nodes_reduced.to_csv("nodes_reduced.csv", index=True)
gdf_edges_reduced.to_csv("edges_reduced.csv", index=True)

with open("nodes_reduced.txt", "w", encoding="utf-8") as f:
    f.write(gdf_nodes_reduced.to_string())

with open("edges_reduced.txt", "w", encoding="utf-8") as f:
    f.write(gdf_edges_reduced.to_string())

print("Saved reduced nodes/edges to nodes_reduced.csv / edges_reduced.csv")

# --- プロット（フルデータ版） ---
fig, ax = plt.subplots(figsize=(8, 8))

# フルデータのエッジとノード
gdf_edges.plot(ax=ax, linewidth=1, edgecolor="red", zorder=2)

if len(gdf_nodes) > 0:
    try:
        gdf_nodes.plot(ax=ax, markersize=6, color="blue", zorder=3)
    except Exception:
        pass

# 表示範囲設定
if len(gdf_edges) > 0:
    xmin, ymin, xmax, ymax = gdf_edges.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_edges.crs, zorder=1)

ax.set_axis_off()
output_path = "node_env.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")

# --- プロット（reduced版） ---
fig, ax = plt.subplots(figsize=(8, 8))

# reduced のエッジとノード
if len(gdf_edges_reduced) > 0:
    gdf_edges_reduced.plot(ax=ax, linewidth=1, edgecolor="red", zorder=2)

if len(gdf_nodes_reduced) > 0:
    try:
        gdf_nodes_reduced.plot(ax=ax, markersize=6, color="blue", zorder=3)
    except Exception:
        pass

# 表示範囲設定
if len(gdf_edges_reduced) > 0:
    xmin, ymin, xmax, ymax = gdf_edges_reduced.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
elif len(gdf_edges) > 0:
    xmin, ymin, xmax, ymax = gdf_edges.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_edges_reduced.crs if len(gdf_edges_reduced) > 0 else gdf_edges.crs, zorder=1)

ax.set_axis_off()
output_path_reduced = "node_env_reduced.png"
fig.savefig(output_path_reduced, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path_reduced}")

# 表示する場合は以下の行のコメントを外してください
# plt.show()


# =============================================================================
# NodeEnv: Q学習用のノード・エッジベース環境クラス
# =============================================================================
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Optional, Tuple


def load_graph_from_csv(nodes_csv="nodes_reduced.csv", edges_csv="edges_reduced.csv",
                        nodes_fallback="nodes.csv", edges_fallback="edges.csv"):
    """CSV からノードとエッジを読み込み、隣接リストを構築する"""
    import os
    # 優先ファイルを試し、なければフォールバック
    npath = nodes_csv if os.path.exists(nodes_csv) else nodes_fallback
    epath = edges_csv if os.path.exists(edges_csv) else edges_fallback

    if not os.path.exists(npath) or not os.path.exists(epath):
        print(f"Warning: CSV files not found. npath={npath}, epath={epath}")
        return None, None, {}

    nodes_df = pd.read_csv(npath, index_col=0, encoding="utf-8")
    edges_df = pd.read_csv(epath, encoding="utf-8")

    # 隣接リスト構築 (u, v カラムを使用)
    adj: Dict[int, List[int]] = {}
    if 'u' in edges_df.columns and 'v' in edges_df.columns:
        for _, row in edges_df.iterrows():
            try:
                u = int(row['u'])
                v = int(row['v'])
            except Exception as e:
                print(f"Warning: Could not parse edge: {row['u']}, {row['v']}, error: {e}")
                continue
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    else:
        print(f"Warning: 'u' or 'v' column not found in edges_df. Columns: {edges_df.columns.tolist()}")

    return nodes_df, edges_df, adj


class NodeEnv:
    """
    ノード間をエッジで移動する Q 学習用環境

    - ノードID（OSM の osmid）を内部で 0..N-1 の state index にマップ
    - actions: 0..(max_degree) の固定長。action < degree → その隣接ノードへ移動、それ以上 → 留まる
    - shelters/disasters: ノードIDまたは state index のリストで指定可能
    """

    def __init__(
        self,
        nodes_df: pd.DataFrame = None,
        edges_df: pd.DataFrame = None,
        adj: Dict[int, List[int]] = None,
        shelters: Optional[List[int]] = None,
        disasters: Optional[List[int]] = None,
        max_steps: int = 100,
        randomize_disasters: bool = False,
        seed: int = 0,
    ):
        random.seed(seed)
        np.random.seed(seed)

        # CSV から読み込まれていなければロード
        if nodes_df is None or edges_df is None or adj is None:
            nodes_df, edges_df, adj = load_graph_from_csv()

        if nodes_df is None or adj is None or len(adj) == 0:
            raise ValueError("ノードまたはエッジデータが見つかりません。先に node_env.py を実行して CSV を生成してください。")

        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.adj_by_nodeid = adj

        # ノードID一覧（DataFrame の index を使用）
        try:
            node_ids = [int(i) for i in nodes_df.index.tolist()]
        except Exception:
            node_ids = list(adj.keys())

        # edges にあるが nodes にないノードを追加
        for nid in list(adj.keys()):
            if nid not in node_ids:
                node_ids.append(nid)

        # nodeid <-> state index マッピング
        self.nodeid_to_state = {nid: i for i, nid in enumerate(node_ids)}
        self.state_to_nodeid = {i: nid for nid, i in self.nodeid_to_state.items()}

        # 隣接リストを state index ベースに変換
        self.neighbors: Dict[int, List[int]] = {}
        maxdeg = 0
        for nid in node_ids:
            nbs = adj.get(nid, [])
            s_nbs = [self.nodeid_to_state[nn] for nn in nbs if nn in self.nodeid_to_state]
            self.neighbors[self.nodeid_to_state[nid]] = s_nbs
            maxdeg = max(maxdeg, len(s_nbs))

        self.n_states = len(node_ids)
        self.n_actions = max(1, maxdeg + 1)  # 最後のアクションは「留まる」

        # shelters / disasters を state に変換
        def _to_states(lst):
            if not lst:
                return []
            out = []
            for v in lst:
                if v in self.nodeid_to_state:
                    out.append(self.nodeid_to_state[v])
                else:
                    try:
                        vi = int(v)
                        if 0 <= vi < self.n_states:
                            out.append(vi)
                    except Exception:
                        pass
            return out

        self.shelters = _to_states(shelters)
        self.disasters = _to_states(disasters)
        self.randomize_disasters = randomize_disasters
        self.max_steps = int(max_steps)

        # ランタイム状態
        self._pos = 0
        self._steps = 0

    def reset(self, start: Optional[int] = None) -> int:
        """環境をリセットし、開始 state を返す"""
        if start is None:
            candidates = [s for s in range(self.n_states) if s not in self.disasters]
            self._pos = random.choice(candidates) if candidates else 0
        else:
            if start in self.nodeid_to_state:
                self._pos = self.nodeid_to_state[start]
            elif 0 <= start < self.n_states:
                self._pos = start
            else:
                self._pos = 0
        self._steps = 0
        return self._pos

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        行動を実行し、(next_state, reward, done, info) を返す
        action < degree: その隣接ノードへ移動
        action >= degree: 留まる
        """
        cur = self._pos
        nbs = self.neighbors.get(cur, [])

        if 0 <= action < len(nbs):
            next_s = nbs[action]
        else:
            next_s = cur  # 留まる

        self._pos = next_s
        self._steps += 1

        done = False
        reward = -1.0

        if next_s in self.shelters:
            done = True
            reward = 100.0
        elif next_s in self.disasters:
            done = True
            reward = -100.0
        elif self._steps >= self.max_steps:
            done = True

        return next_s, reward, done, {}

    def neighbors_of(self, state: int) -> List[int]:
        """指定 state の隣接 state リストを返す"""
        return self.neighbors.get(state, [])

    def nodeid_of(self, state: int) -> Optional[int]:
        """state index → 元のノードID"""
        return self.state_to_nodeid.get(state)

    def state_of(self, nodeid: int) -> Optional[int]:
        """ノードID → state index"""
        return self.nodeid_to_state.get(nodeid)

    def render(self):
        """現在の状態を表示"""
        print(f"NodeEnv: {self.n_states} nodes, {self.n_actions} actions")
        print(f"pos(state)={self._pos}, nodeid={self.nodeid_of(self._pos)}")
        print(f"shelters(states)={self.shelters}")
        print(f"disasters(states)={self.disasters}")