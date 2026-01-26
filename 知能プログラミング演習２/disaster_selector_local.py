from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

def select_disaster_nodes(osm_nodes, n_disasters=1):
    """
    建物に近いほど災害が起きやすい
    1か所だけ重み付きランダムで選ぶ
    """

    scores = []

    for n in osm_nodes:
        dist = n.get("dist_to_building", 9999)  # 建物からの距離

        # 🔥 建物に近いほど値が大きくなるスコア
        score = 1 / (dist + 20)

        scores.append(score)

    scores = np.array(scores)

    # 確率に変換
    probs = scores / scores.sum()

    # 🎯 1か所だけ抽選（replace=Falseで重複なし）
    chosen_index = np.random.choice(
        len(osm_nodes),
        size=1,
        replace=False,
        p=probs
    )[0]

    # ノードIDをリストで返す（今のrun_select_disaster.pyと互換性を保つため）
    return [osm_nodes[chosen_index]["id"]]
