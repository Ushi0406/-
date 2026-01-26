import argparse
import numpy as np
import os
import time #aaa

# ===== LLM 避難指示生成の準備 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model_llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

#避難指示生成仮
def generate_evacuation_text(path, env):
    coords = [(p % env.w, p // env.w) for p in path]

    prompt = f"""
あなたは防災ナビゲーションAIです。
以下は避難者の移動経路です。

開始地点から避難所までの避難指示を日本語でわかりやすく説明してください。

移動経路:
{coords}

避難指示:
"""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model_llm.generate(input_ids, max_length=150)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text
    
# print_grid_step: コンソールにグリッドの現在状態を描画する補助関数
# - env: GridEnv インスタンス（グリッド幅、高さ、災害・避難所位置を参照）
# - current_pos: 現在の状態インデックス（0..w*h-1）
# - path_so_far: これまで通った状態インデックスのリスト（経路表示）
def print_grid_step(env, current_pos, path_so_far=None):
    # '.' = 空セル, 'X' = 災害, 'S' = 避難所, 'A' = エージェント, 数字 = 経路インデックスの末尾桁
    grid = [['.' for _ in range(env.w)] for _ in range(env.h)]

    # 災害セルを 'X' で描画
    for d in env.disasters:
        x, y = d % env.w, d // env.w
        grid[y][x] = 'X'

    # 避難所を 'S' で描画
    for s in env.shelters:
        x, y = s % env.w, s // env.w
        grid[y][x] = 'S'

    # 経路を数字で描画（既にS/Xなら上書きしない）
    if path_so_far:
        for i, pos in enumerate(path_so_far):
            x, y = pos % env.w, pos // env.w
            if grid[y][x] in ('S', 'X'):
                continue
            grid[y][x] = str(i % 10)

    # エージェント位置を 'A' で表示（経路表示より優先）
    x, y = current_pos % env.w, current_pos // env.w
    grid[y][x] = 'A'

    # 行ごとに出力
    for row in grid:
        print(''.join(row))


# greedy_run: 学習済 Q テーブルに従い貪欲に1ステップずつ行動して表示する
# - env: GridEnv インスタンス
# - Q: shape (n_states, n_actions) の Q テーブル
# - start: 開始状態インデックス（None なら env.reset() のデフォルト）
# - max_steps: 最大ステップ数（安全策）
# - delay: 各ステップ表示の待ち時間（秒）
# - clear: True なら Windows のコンソールをクリアしてアニメーション風に表示
def greedy_run(env, Q, start=None, max_steps=200, delay=0.5, clear=True):
    # 環境をリセットして開始位置を取得
    s = env.reset(start=start)
    path = [s]
    done = False

    for step in range(max_steps):
        # 画面クリア（Windows の場合）
        if clear:
            os.system('cls')

        # 現在ステップと位置を表示
        print(f"Step {step}, pos={(s % env.w, s // env.w)}")
        print_grid_step(env, s, path)

        # 見やすさのため一時停止
        time.sleep(delay)

        # Q テーブルに基づく貪欲選択（最大Qの行動）
        a = int(np.argmax(Q[s]))

        # 行動を適用して次状態・報酬・終了フラグを取得
        s, r, done, _ = env.step(a)
        path.append(s)

        # 終了したら最終表示してループを抜ける
        if done:
            if clear:
                os.system('cls')
            print(f"Step {step+1}, pos={(s % env.w, s // env.w)} (Reached shelter or terminal)")
            print_grid_step(env, s, path)
            break

    return path, done


# print_path: 最終的な経路を静的に表示する（アニメーション後の確認用）
def print_path(env, path):
    grid = [['.' for _ in range(env.w)] for _ in range(env.h)]

    for d in env.disasters:
        x, y = d % env.w, d // env.w
        grid[y][x] = 'X'
    for s in env.shelters:
        x, y = s % env.w, s // env.w
        grid[y][x] = 'S'

    for i, pos in enumerate(path):
        x, y = pos % env.w, pos // env.w
        if grid[y][x] in ('S', 'X'):
            continue
        grid[y][x] = str(i % 10)

    for row in grid:
        print(''.join(row))


if __name__ == '__main__':
    # コマンドライン引数（デフォルトや説明を設定）
    p = argparse.ArgumentParser(description='学習済 Q テーブルで貪欲実行して経路を表示するデモ')
    p.add_argument('--model', type=str, default='model/q_table.npy', help='学習済 Q テーブルのパス')
    p.add_argument('--width', type=int, default=5, help='グリッド幅')
    p.add_argument('--height', type=int, default=5, help='グリッド高さ')
    p.add_argument('--start', type=str, default=None, help='開始座標 x,y (例: --start 0,0)')
    p.add_argument('--seed', type=int, default=0, help='乱数シード（災害ランダム化などに使用）')
    p.add_argument('--delay', type=float, default=0.5, help='各ステップ表示の待ち時間（秒）')
    args = p.parse_args()

    # GridEnv 側で避難所・災害位置を管理しているため、ここでは幅・高さ等のみ渡す
    env = GridEnv(width=args.width, height=args.height, randomize_disasters=False, max_steps=200, seed=args.seed)

    # 学習済 Q を読み込む。ファイルが無ければ小さな乱数Qで代替（デバッグ用）
    if os.path.exists(args.model):
        Q = np.load(args.model)
    else:
        print(f"モデルが見つかりません: {args.model}  → ランダムQで代替します")
        Q = np.random.rand(env.n_states, env.n_actions) * 0.01

    # 開始位置を x,y 形式の引数から計算（None のままなら env.reset のデフォルト）
    start_pos = None
    if args.start:
        x, y = map(int, args.start.split(','))
        start_pos = y * args.width + x

    # 貪欲実行（1ステップずつ表示）
    path, done = greedy_run(env, Q, start=start_pos, delay=args.delay)
    print('Reached shelter:', done)
    print_path(env, path)

#以下避難指示仮
path, done = greedy_run(env, Q, start=start_pos, delay=args.delay)
print('Reached shelter:', done)
print_path(env, path)

print("\n避難ナビゲーションAIからの指示")
evac_text = generate_evacuation_text(path, env)
print(evac_text)
