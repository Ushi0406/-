from node_env import load_reduced_nodes, plot_with_selected
from disaster_selector_local import select_disaster_nodes

# ① ノード読み込み
nodes = load_reduced_nodes()

print(f"LLMに渡すノード数: {len(nodes)}")
print(nodes[:3])

# ② LLMで災害ノード選択
disaster_ids = select_disaster_nodes(
    nodes,
    n_disasters=3
)

'''
# ③ 座標つきで表示
id_to_node = {n["id"]: n for n in nodes}

print("\n=== 災害ノード（LLM選択）===")
for nid in disaster_ids:
    n = id_to_node.get(nid)
    if n:
        print(
            f"id={nid}, "
            f"x={n['x']:.1f}, "
            f"y={n['y']:.1f}"
        )
'''

id_to_node = {n["short_id"]: n for n in nodes}

print("\n=== 災害ノード（LLM選択）===")
for sid in disaster_ids:
    n = id_to_node.get(sid)
    if n:
        print(f"short_id={sid}, osm_id={n['osm_id']}, x={n['x']:.1f}, y={n['y']:.1f}")

# ④ 地図上に災害ノードを可視化
selected_osm_ids = [id_to_node[sid]["osm_id"] for sid in disaster_ids if sid in id_to_node]
plot_with_selected(selected_osm_ids, out_path="grid_env_disaster.png")