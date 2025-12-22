
import numpy as np

class GridEnv:
    """
    単純なグリッド環境。
    - 状態: 0 .. w*h-1（行優先、state = y*w + x）
    - 行動: 0=上, 1=右, 2=下, 3=左
    - 災害セルに入ると大きな罰 (-100) でエピソード終了
    - 避難所に入ると大きな報酬 (+100) でエピソード終了
    - 通常移動は -1 の小さな負報酬（時間コスト）
    """

    def __init__(self, width=5, height=5, shelters=None, disasters=None,
                 n_disasters=None, randomize_disasters=False, max_steps=200, seed=None):
        self.w = width
        self.h = height
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)

        # デフォルト避難所: 右下（gridのインデックス）
        if shelters is None:
            self.shelters = [ (self.h - 1) * self.w + (self.w - 1) ]
        else:
            self.shelters = list(shelters)

        # デフォルト災害候補座標（必要に応じて変更可）
        default_coords = [(1,1),(2,1),(1,2),(2,2),(0,2),(3,1),(3,2),(2,3)]
        coords_in_grid = [(x,y) for (x,y) in default_coords if x < self.w and y < self.h]
        default_idxs = [y*self.w + x for (x,y) in coords_in_grid]

        # 災害位置の初期化: 明示的に渡された場合はそれを使い、渡されなければデフォルトから選択
        if disasters is None:
            if n_disasters is None:
                n_disasters = min(3, len(default_idxs))
            if randomize_disasters:
                # 重複なしでランダム選択
                self.disasters = list(self.rng.choice(default_idxs, size=n_disasters, replace=False))
            else:
                # デフォルトリストの先頭を使う
                self.disasters = default_idxs[:n_disasters]
        else:
            self.disasters = list(disasters)

        # 状態数・行動数の設定（プロパティに変更）
        #        self.n_states = self.w * self.h
        #        self.n_actions = 4  # up,right,down,left
        # n_states / n_actions はプロパティで計算するためここでは設定しない

        # 初期化（開始位置・ステップ数など）
        self.reset()

    @property
    def n_states(self):
        """状態数: w*h を動的に返すプロパティ"""
        return self.w * self.h

    @property
    def n_actions(self):
        """行動数: 上・右・下・左 を固定で返すプロパティ"""
        return 4

    # 環境をリセットして開始状態を返す
    # start: 指定された状態インデックスを開始位置にする（None の場合左上(0,0)）
    def reset(self, start=None):
        self.agent = 0 if start is None else start
        self.steps = 0
        return self.agent

    # 行動を実行して (next_state, reward, done, info) を返す
    def step(self, action):
        x = self.agent % self.w
        y = self.agent // self.w
        if action == 0 and y > 0:      # up
            y -= 1
        elif action == 1 and x < self.w - 1:  # right
            x += 1
        elif action == 2 and y < self.h - 1:  # down
            y += 1
        elif action == 3 and x > 0:    # left
            x -= 1
        next_state = y * self.w + x
        self.agent = next_state
        self.steps += 1

        # 災害セルに到達したら大きな罰で終了
        if self.agent in self.disasters:
            return self.agent, -100.0, True, {}
        # 避難所に到達したら大きな報酬で終了
        if self.agent in self.shelters:
            return self.agent, 100.0, True, {}
        # 最大ステップ到達で終了（タイムアウト）
        if self.steps >= self.max_steps:
            return self.agent, -1.0, True, {}

        # 通常の移動は小さな負報酬（時間コスト）
        return self.agent, -1.0, False, {}

    def render(self, show_disasters=True):
        grid = np.full((self.h, self.w), '.', dtype=str)
        for s in self.shelters:
            x, y = self._pos_to_xy(s)
            grid[y, x] = 'S'
        if show_disasters:
            for d in self.disasters:
                x, y = self._pos_to_xy(d)
                # 災害が避難所と重なることはない
                grid[y, x] = 'X'
        ax, ay = self._pos_to_xy(self.agent_pos)
        grid[ay, ax] = 'A'
        lines = [''.join(row) for row in grid]
        print('\n'.join(lines))
