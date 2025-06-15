# 🏠 Smart Home Energy Optimization using Reinforcement Learning

A Reinforcement Learning (RL) project that simulates intelligent control of home appliances (light, fan, heater) to minimize energy consumption and maximize efficiency using Q-learning.

## 📌 Project Highlights

- **Q-learning Agent**: Learns optimal actions based on state features like time, temperature, and presence.
- **Custom SmartHome Environment**: Simulates real-world energy scenarios.
- **MLflow Integration**: Tracks experiments, hyperparameters, and rewards.
- **Streamlit App**: Web app for simulating and visualizing agent behavior.
- **Reward Optimization**: Agent trained over multiple configurations to reach optimal energy usage patterns.

---

## 📊 Problem Statement

In smart homes, appliances often stay on unnecessarily, leading to energy wastage. This project builds an RL agent that **learns when to turn appliances ON or OFF** based on the environment, aiming to:

- Reduce energy usage
- Maintain comfort
- Learn from experience

---

## 🚀 Technologies Used

| Tool         | Purpose                                      |
|--------------|----------------------------------------------|
| **Python**   | Core programming language                    |
| **Q-learning** | Reinforcement Learning algorithm           |
| **MLflow**   | Experiment tracking                          |
| **Streamlit**| Visualization and simulation UI              |
| **Matplotlib/Pandas** | Data analysis and plotting         |

---

## 🧠 How It Works

1. **Environment** simulates room temperature, device states, etc.
2. **Agent** chooses an action based on the current state (explore/exploit).
3. **Reward** is calculated (e.g., energy saved vs comfort compromised).
4. **Q-table** is updated to improve future decisions.
5. **MLflow** logs each run's performance for comparison.
6. **Streamlit App** allows user to run simulations interactively.

---

## 📂 Project Structure


