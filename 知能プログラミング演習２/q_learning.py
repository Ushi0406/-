# q_learning.py
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.97,
                 epsilon=1.0, epsilon_min=0.10, epsilon_decay=0.9997):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.Q = defaultdict(float)

    def choose_action(self, env, s):
        actions = env.actions(s)
        if not actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(actions)

        qs = [self.Q[(s, a)] for a in actions]
        m = max(qs)
        best = [a for a, q in zip(actions, qs) if q == m]
        return random.choice(best)  # 同点はランダム

    def update(self, env, s, a, r, s2, done):
        if done:
            target = r
        else:
            a2s = env.actions(s2)
            target = r + (self.gamma * max(self.Q[(s2, a2)] for a2 in a2s)) if a2s else r
        self.Q[(s, a)] += self.alpha * (target - self.Q[(s, a)])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
