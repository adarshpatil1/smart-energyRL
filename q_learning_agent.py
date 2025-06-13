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

    def get_state_key(self, state):
        return tuple(np.round(state, 2))  # Discretize state for key

    def choose_action(self, state):
        key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[key])  # Exploit

    def update(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        current_q = self.q_table[key][action]
        max_next_q = np.max(self.q_table[next_key])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        self.q_table[key][action] = new_q