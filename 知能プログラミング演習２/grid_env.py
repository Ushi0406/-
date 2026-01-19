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
plt.show()
