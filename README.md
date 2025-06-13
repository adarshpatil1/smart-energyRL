# Smart Energy RL Optimization 🔋🏠

This project simulates a smart home energy system and uses **Reinforcement Learning (RL)** to optimize the usage of home devices (heater, light, fan) to reduce energy costs while maintaining user comfort.

## 🚀 Project Overview

In a smart home setting, devices can be turned on/off based on external temperature and time of day. The goal is to:
- Minimize energy usage cost
- Avoid comfort penalties (e.g., cold nights without heating or dark evenings without lights)
- Learn control policies automatically using Q-learning

We used **Q-Learning**, a model-free RL algorithm, to learn when to turn each device on/off for energy and comfort optimization.

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

We used it to:
- Compare different hyperparameter settings
- Log reward per episode
- Save the best-performing Q-table for reproducibility

---

## 🐳 Why Docker?

**Why Docker?**
Docker ensures:
- Consistent environment across systems
- Easier deployment and sharing
- Dependency isolation

In a real-world setup, this smart control system could run on a Raspberry Pi or cloud server — Docker helps us make it portable.

---

## 🌐 FastAPI (or Streamlit)

While this project is RL-focused, in the next phase:
- **FastAPI** can expose the trained model as a REST API, allowing external apps to query the best action (for deployment).
- **Streamlit** can visualize training logs, live device behavior, and reward progress for better explainability.

These tools are used in professional MLOps pipelines to bridge models with real-time systems and users.

---

## 📁 Project Structure

