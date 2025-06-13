import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.05, discount=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.n_actions = n_actions

    def _get_state(self):
        return (
        int(self.devices["heater"]),
        int(self.devices["light"]),
        int(self.devices["fan"]),
        round(self.temp_outside),  # or int(...)
        self.hour
        )

    def choose_action(self, state):
        key = tuple(state)
        if key in self.q_table:
            return np.argmax(self.q_table[key])  # Exploit
        else:
            return np.random.choice(self.n_actions)  # Safe fallback

    def update(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        current_q = self.q_table[key][action]
        max_next_q = np.max(self.q_table[next_key])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        self.q_table[key][action] = new_q