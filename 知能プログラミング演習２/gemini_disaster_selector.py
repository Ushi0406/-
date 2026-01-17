import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.0-flash-001"


def select_disaster_nodes(osm_nodes, n_disasters=3):
    """
    OSMノード一覧を与え、Geminiに災害ノードを選ばせる
    戻り値: ノードIDのリスト
    """

    node_desc = "\n".join(
        [f"- id:{n['id']} (x={n['x']}, y={n['y']}), type={n['type']}"
         for n in osm_nodes]
    )

    prompt = f"""
以下は避難シミュレーション用の地図ノードです。
洪水・火災・倒壊の危険が高そうな場所を {n_disasters} 個選んでください。

条件:
- river_near, narrow_road, intersection は危険度が高い
- school や shelter は避ける
- 出力はノードIDのみ（カンマ区切り）

ノード一覧:
{node_desc}

出力例:
2,4
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = response.text.strip()
    ids = [int(x) for x in text.split(",") if x.strip().isdigit()]
    return ids

def nodes_to_disasters(osm_nodes, selected_ids, width):
    """
    Geminiが選んだノードID → GridEnv用 disasters（state index）
    """
    id_to_node = {n["id"]: n for n in osm_nodes}
    disasters = []

    for nid in selected_ids:
        node = id_to_node[nid]
        state = node["y"] * width + node["x"]
        disasters.append(state)

    return disasters
