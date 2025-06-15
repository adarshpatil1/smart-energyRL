import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

from smart_home_env import SmartHomeEnv
from q_learning_agent import QLearningAgent

# Load the trained Q-table
@st.cache_data
def load_q_table(path="q_table_C_4.pkl"):
    with open(path, "rb") as f:
        q_table = pickle.load(f)
    return {normalize_key(k): v for k, v in q_table.items()}

def normalize_key(key):
    return tuple(int(x) if isinstance(x, (np.integer, np.int64, np.int32)) else
                 float(x) if isinstance(x, (np.floating, np.float64, np.float32)) else x
                 for x in key)

# Run simulation using Q-table
def run_simulation(q_table, num_steps=200):
    env = SmartHomeEnv()
    agent = QLearningAgent(n_actions=env.action_space.n)
    agent.q_table = q_table

    state = tuple(env.reset())  # Convert initial state to tuple
    total_reward = 0
    rewards = []
    actions = []
    states = [state]  # Collect initial state

    for _ in range(num_steps):
        action = agent.choose_action(state)
        actions.append(action)

        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)  # Convert next state to tuple
        rewards.append(reward)
        total_reward += reward
        states.append(next_state)  # Append next state

        if done:
            break
        state = next_state

    return rewards, actions, states, total_reward


# --- Streamlit UI ---
st.title("🔋 Smart Energy Q-Learning Agent")
st.markdown("Simulate a smart energy environment using a pre-trained Q-table.")

q_table = load_q_table()

# Optional Check 1: Print a few keys from Q-table to verify structure
st.write("Sample Q-table keys)")
for i, key in enumerate(q_table.keys()):
    st.write(key)
    if i >= 2:
        break


# User input

steps = st.slider("Simulation Steps (24 = 1 day)", min_value=10, max_value=500, value=24, help="Default is 24 steps (1 day simulation)")
st.markdown("**Note:** The main metric is the total reward for one day (24 steps). For best comparison with MLflow, use 24 steps.")

if st.button("Run Simulation"):
    rewards, actions, states, total_reward = run_simulation(q_table, num_steps=steps)

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
    if st.checkbox("Show Sample Q-table keys"):
        sample_keys = list(q_table.keys())[:3]
        st.code(f"Sample Q-table keys:\n\n{sample_keys}", language='python')
    
