# 単一避難者 RL（Q学習）サンプル

使い方（Windows のコマンドプロンプト / PowerShell で実行）：

1. トレーニング

```
python train_q_learning.py --width 5 --height 5 --n_disasters 3 --shelters "4,4" --episodes 5000
```

- `--shelters` は `x,y` の形式。複数あるときはセミコロンで区切る（例: "4,4;0,0"）。
- トレーニング後、`model/q_table.npy` が保存されます。

2. 学習済みモデルでデモ実行

```
python run_demo.py --model model/q_table.npy --width 5 --height 5 --n_disasters 3 --shelters "4,4"
```

- 標準出力にグリッドを表示します。`S` が避難所、`X` が災害、数字は経路上のステップ（0-9の繰り返し）です。

調整パラメータ：学習率（`--alpha`），割引率（`--gamma`），探索率（`--epsilon`），エピソード数（`--episodes`）。

注意点：このサンプル実装では災害マップは初期化時にランダムで配置され、トレーニング中は固定されます（`randomize_disasters=False`）。複数の災害配置に対して一般化させたい場合は`train_q_learning.py`を改良してください。

node_env.py：地図データ作成

disaster_selector_local.py：災害発生場所を決める(建物との距離に近いほど確率高い)

run_select_disaster.py：シミュレーション実行用



