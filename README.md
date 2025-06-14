# Smart Energy RL Optimization 🔋🏠

This project simulates a smart home energy system and uses **Reinforcement Learning (RL)** to optimize the usage of home devices (heater, light, fan) to reduce energy costs while maintaining user comfort.

## 🚀 Project Overview

In a smart home setting, devices can be turned on/off based on external temperature and time of day. The goal is to:
- Minimize energy usage cost
- Avoid comfort penalties (e.g., cold nights without heating or dark evenings without lights)
- Learn control policies automatically using Q-learning

 Here I used **Q-Learning**, a model-free RL algorithm, to learn when to turn each device on/off for energy and comfort optimization.

## 🧠 Why Q-Learning?

Q-Learning is simple yet powerful for environments with discrete state/action spaces like our smart home. It doesn’t require a simulation model and learns through exploration, making it ideal for:
- Simple to moderate control problems
- Fast convergence in low-dimensional state spaces

---

## 📊 MLflow for Experiment Tracking

**Why MLflow?**
MLflow helps track:
- Model parameters (learning rate, discount factor, epsilon)
- Metrics (total reward)
- Artifacts (Q-table)

Here I used it to:
- Compare different hyperparameter settings
- Log reward per episode
- Save the best-performing Q-table for reproducibility

---
