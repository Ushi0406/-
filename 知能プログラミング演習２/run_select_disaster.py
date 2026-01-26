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

id_to_node = {n["id"]: n for n in nodes}

print("\n=== 災害ノード（LLM選択）===")
for nid in disaster_ids:
    n = id_to_node.get(nid)
    if n:
        print(f"id={nid}, x={n['x']:.1f}, y={n['y']:.1f}")

# ④ 地図上に災害ノードを可視化
selected_osm_ids = [nid for nid in disaster_ids if nid in id_to_node]
plot_with_selected(selected_osm_ids, out_path="grid_env_disaster.png")
