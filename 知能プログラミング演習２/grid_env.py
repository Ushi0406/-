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

# 追加: ノードとエッジをテキスト（CSV と TXT）で出力
nodes_csv = "nodes.csv"
edges_csv = "edges.csv"
nodes_txt = "nodes.txt"
edges_txt = "edges.txt"

# CSV に保存（geometry 列は WKB/WKT ではなく文字列化されます）
gdf_nodes.to_csv(nodes_csv, index=True)
gdf_edges.to_csv(edges_csv, index=True)

# 見やすいテキスト出力（全行を書き出す）
with open(nodes_txt, "w", encoding="utf-8") as f:
    f.write(gdf_nodes.to_string())

with open(edges_txt, "w", encoding="utf-8") as f:
    f.write(gdf_edges.to_string())

print(f"Saved nodes to {nodes_csv} and {nodes_txt}")
print(f"Saved edges to {edges_csv} and {edges_txt}")

fig, ax = plt.subplots(figsize=(8, 8))

# ④ 道路ネットワーク描画
gdf_edges.plot(
    ax=ax,
    linewidth=1,
    edgecolor="red",
    zorder=2
)

# ⑤ 表示範囲設定
xmin, ymin, xmax, ymax = gdf_edges.total_bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ⑥ OpenStreetMap を背景に追加
ctx.add_basemap(
    ax,
    source=ctx.providers.OpenStreetMap.Mapnik,
    crs=gdf_edges.crs,
    zorder=1
)

ax.set_axis_off()
# 画像として保存
output_path = "grid_env.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")

# 表示する場合は以下の行のコメントを外してください
plt.show()
