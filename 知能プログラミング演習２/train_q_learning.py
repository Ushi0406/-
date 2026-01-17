import argparse
import numpy as np
import os
from grid_env import GridEnv
from gemini_disaster_selector import (
    select_disaster_nodes,
    nodes_to_disasters
)


def train(args):
    #仮のノード変更する
    osm_nodes = [
        {"id": 1, "x": 1, "y": 1, "type": "a"},
        {"id": 2, "x": 2, "y": 1, "type": "b"},
        {"id": 3, "x": 3, "y": 2, "type": "c"},
        {"id": 4, "x": 0, "y": 2, "type": "d"},
        {"id": 5, "x": 3, "y": 1, "type": "e"},
        {"id": 6, "x": 3, "y": 2, "type": "f"},
        {"id": 7, "x": 2, "y": 3, "type": "g"},
    ]

    shelters = []
    if args.shelters:
        for s in args.shelters.split(';'):
            x, y = map(int, s.split(','))
            shelters.append(y * args.width + x)

    # --- Gemini による災害ノード選択 ---
    selected_ids = select_disaster_nodes(
        osm_nodes,
        n_disasters=args.n_disasters
    )

    # ノードID → GridEnv用 disasters
    disasters = nodes_to_disasters(
        osm_nodes,
        selected_ids,
        width=args.width
    )


    '''env = GridEnv(width=args.width, height=args.height, shelters=shelters,
                  n_disasters=args.n_disasters, max_steps=args.max_steps, randomize_disasters=False, seed=args.seed)'''
    env = GridEnv(
        width=args.width,
        height=args.height,
        shelters=shelters,
        disasters=disasters,   #Geminiが選んだ災害
        max_steps=args.max_steps,
        seed=args.seed
    )


    n_states = env.n_states
    n_actions = env.n_actions
    Q = np.zeros((n_states, n_actions))

    alpha = args.alpha
    gamma = args.gamma
    eps = args.epsilon

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < eps:
                a = np.random.randint(n_actions)
            else:
                a = int(np.argmax(Q[state]))
            next_s, r, done, _ = env.step(a)
            Q[state, a] = Q[state, a] + alpha * (r + gamma * np.max(Q[next_s]) - Q[state, a])
            state = next_s
        # epsilon decay
        if ep % 100 == 0 and ep > 0 and args.epsilon_decay:
            eps = max(args.min_epsilon, eps * args.epsilon_decay)
        if (ep+1) % (args.episodes//10 if args.episodes>=10 else 1) == 0:
            print(f"Episode {ep+1}/{args.episodes}")

    os.makedirs(args.out_dir, exist_ok=True)
    qpath = os.path.join(args.out_dir, 'q_table.npy')
    np.save(qpath, Q)
    print('Saved Q-table to', qpath)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=5)
    p.add_argument('--height', type=int, default=5)
    p.add_argument('--n_disasters', type=int, default=3)
    p.add_argument('--shelters', type=str, default='4,4', help='複数はセミコロン区切りのx,y例: "4,4;0,0"')
    p.add_argument('--episodes', type=int, default=5000)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--epsilon', type=float, default=0.2)
    p.add_argument('--epsilon_decay', type=float, default=0.995)
    p.add_argument('--min_epsilon', type=float, default=0.01)
    p.add_argument('--max_steps', type=int, default=100)
    p.add_argument('--out_dir', type=str, default='model')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    train(args)
