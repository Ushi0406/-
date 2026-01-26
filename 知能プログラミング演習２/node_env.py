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

# --- プロット ---
fig, ax = plt.subplots(figsize=(8, 8))

# エッジは reduced を使用（存在しない場合はフル）
plot_edges = gdf_edges_reduced if len(gdf_edges_reduced) > 0 else gdf_edges
plot_edges.plot(ax=ax, linewidth=1, edgecolor="red", zorder=2)

# ノードは reduced を使用
if len(gdf_nodes_reduced) > 0:
    try:
        gdf_nodes_reduced.plot(ax=ax, markersize=6, color="blue", zorder=3)
    except Exception:
        pass

# 表示範囲は reduced エッジの範囲（なければフル）
if len(plot_edges) > 0:
    xmin, ymin, xmax, ymax = plot_edges.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=plot_edges.crs, zorder=1)

ax.set_axis_off()
output_path = "grid_env.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")

# 表示する場合は以下の行のコメントを外してください
plt.show()

def load_reduced_nodes():
    """
    reduced ノードを Gemini 用の辞書配列として返す
    """
    nodes = []

    for idx, row in gdf_nodes_reduced.iterrows():
        nodes.append({
            "id": int(idx),                  # OSM node id
            "x": float(row.geometry.x),      # Web Mercator x
            "y": float(row.geometry.y),      # Web Mercator y
            "type": "intersection"           # 今回は固定
        })

    return nodes

def plot_with_selected(selected_osm_ids, out_path="grid_env_selected.png"):
    """
    選ばれたOSMノードIDを地図上で強調表示して保存する
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # エッジ
    plot_edges = gdf_edges_reduced if len(gdf_edges_reduced) > 0 else gdf_edges
    plot_edges.plot(ax=ax, linewidth=1, edgecolor="red", zorder=2)

    # ノード（全体）
    gdf_nodes_reduced.plot(ax=ax, markersize=20, color="blue", zorder=3)

    # 選ばれたノードだけ強調（黄色）
    sel = gdf_nodes_reduced[gdf_nodes_reduced.index.isin(selected_osm_ids)]
    if len(sel) > 0:
        sel.plot(ax=ax, markersize=80, color="yellow", edgecolor="black", zorder=4)

        # ラベル（OSM IDの末尾だけ表示）
        for osm_id, row in sel.iterrows():
            x = row.geometry.x
            y = row.geometry.y
            ax.text(x, y, str(osm_id)[-4:], fontsize=10, color="black", zorder=5)

    # 範囲
    if len(plot_edges) > 0:
        xmin, ymin, xmax, ymax = plot_edges.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=plot_edges.crs, zorder=1)
    ax.set_axis_off()

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved selected plot to {out_path}")
    plt.close(fig)
