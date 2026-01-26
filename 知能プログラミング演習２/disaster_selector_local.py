from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

def select_disaster_nodes(osm_nodes, n_disasters=3):
    """
    LLM（ローカル）に災害ノードを選ばせる
    osm_nodes: [{"id":int,"x":float,"y":float,"type":str}, ...]
    戻り値: ノードIDのリスト
    """

    # ノード一覧をテキスト化
    node_desc = "\n".join(
        [f"id:{n['id']} (x={n['x']:.1f}, y={n['y']:.1f}), type={n['type']}"
         for n in osm_nodes]
    )

    prompt = f"""
以下は避難シミュレーション用の地図ノードです。
洪水・火災・倒壊の危険が高そうな場所を {n_disasters} 個選んでください。

条件:
- 出力はノードIDのみ（カンマ区切り）

ノード一覧:
{node_desc}

出力例:
"""

    # トークン化
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)

    # LLM生成
    outputs = model.generate(
        input_ids,
        max_length=128,
        do_sample=False
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("LLM output:", output_text)

    # 数字だけ抽出
    ids = [int(x) for x in re.findall(r'\d+', output_text)]

    return ids[:n_disasters]