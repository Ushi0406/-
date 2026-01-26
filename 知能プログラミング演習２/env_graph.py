# env_graph.py
import networkx as nx

class GraphEvacuationEnv:
    def __init__(self, G, start, goal,
                 risk_weight=15.0,
                 step_penalty=1.0,
                 goal_reward=5000.0,
                 revisit_penalty=10.0,
                 backtrack_penalty=20.0,
                 shaping_scale=5.0):
        self.G = G
        self.start = int(start)
        self.goal = int(goal)

        self.risk_weight = float(risk_weight)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.revisit_penalty = float(revisit_penalty)
        self.backtrack_penalty = float(backtrack_penalty)
        self.shaping_scale = float(shaping_scale)

        # ゴールまでの最短距離 d(s) を事前計算（distance属性でDijkstra）
        # 到達不能は無限大にする
        self.dist_to_goal = {n: float("inf") for n in self.G.nodes()}
        try:
            d = nx.single_source_dijkstra_path_length(self.G, self.goal, weight="distance")
            self.dist_to_goal.update(d)
        except Exception:
            pass

        self.state = None
        self.prev_state = None
        self.visited = set()

    def reset(self):
        self.state = self.start
        self.prev_state = None
        self.visited = {self.start}
        return self.state

    def actions(self, s):
        return list(self.G.neighbors(s))

    def step(self, a):
        s = self.state
        s2 = int(a)

        if not self.G.has_edge(s, s2):
            return s, -200.0, False, {"invalid": True}

        data = self.G[s][s2]
        dist = float(data.get("distance", 1.0))
        risk = float(data.get("risk", 1.0))
        blocked = bool(data.get("blocked", False))
        if blocked:
            return s, -500.0, False, {"blocked": True}

        # 基本コスト
        r = -(dist + self.risk_weight * risk) - self.step_penalty

        # ★ shaping：ゴールに近づいたら加点
        # d(s) - d(s2) が正なら近づいた
        ds = self.dist_to_goal.get(s, float("inf"))
        ds2 = self.dist_to_goal.get(s2, float("inf"))
        if ds < float("inf") and ds2 < float("inf"):
            r += self.shaping_scale * (ds - ds2)

        # ループ・後戻り罰
        if s2 in self.visited:
            r -= self.revisit_penalty
        if self.prev_state is not None and s2 == self.prev_state:
            r -= self.backtrack_penalty

        # 状態更新
        self.prev_state = s
        self.state = s2
        self.visited.add(s2)

        done = (s2 == self.goal)
        if done:
            r += self.goal_reward

        return s2, r, done, {}
