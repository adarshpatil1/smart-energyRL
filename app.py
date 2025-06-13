import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

from smart_home_env import SmartHomeEnv
from q_learning_agent import QLearningAgent

# Load the trained Q-table
@st.cache_data
def load_q_table(path="best_q_table.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

# Run simulation using Q-table
def run_simulation(q_table, num_steps=200):
    env = SmartHomeEnv()
    agent = QLearningAgent(n_actions=env.action_space.n)
    agent.q_table = q_table

    state = env.reset()
    total_reward = 0
    rewards = []
    actions = []

    for _ in range(num_steps):
        action = agent.choose_action(state)
        actions.append(action)

        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        total_reward += reward

        if done:
            break
        state = next_state

    return rewards, actions, total_reward

# --- Streamlit UI ---
st.title("🔋 Smart Energy Q-Learning Agent")
st.markdown("Simulate a smart energy environment using a pre-trained Q-table.")

q_table = load_q_table()

# User input
steps = st.slider("Simulation Steps", 10, 200, 50)

if st.button("Run Simulation"):
    rewards, actions, total_reward = run_simulation(q_table, num_steps=steps)

    st.success(f"✅ Simulation Completed - Total Reward: {total_reward}")
    
    # Plot rewards
    fig, ax = plt.subplots()
    ax.plot(rewards, marker='o')
    ax.set_title("Reward per Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

    # Show actions taken
    st.markdown("### Actions Taken")
    st.write(actions)
