import mlflow
from smart_home_env import SmartHomeEnv
from q_learning_agent import QLearningAgent
import pickle

hyperparams_list = [
    {"lr": 0.1, "gamma": 0.95, "epsilon": 0.3},
    {"lr": 0.05, "gamma": 0.99, "epsilon": 0.2},
    {"lr": 0.01, "gamma": 0.90, "epsilon": 0.1},
    {"lr": 0.2, "gamma": 0.85, "epsilon": 0.4},
    {"lr": 0.15, "gamma": 0.9, "epsilon": 0.25},
]

def run_experiment(run_id, config):
    env = SmartHomeEnv()
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=config["lr"],
        discount=config["gamma"],
        epsilon=config["epsilon"]
    )

    num_episodes = 500
    mlflow.set_experiment("SmartHome_RL")

    with mlflow.start_run(run_name=f"Combo_{run_id}"):
        mlflow.log_params(config)

        for episode in range(num_episodes):
            state = env.reset()  # Already a tuple of ints
            total_reward = 0
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            mlflow.log_metric("reward", total_reward, step=episode)

        with open(f"q_table_{run_id}.pkl", "wb") as f:
            pickle.dump(dict(agent.q_table), f)
            mlflow.log_artifact(f"q_table_{run_id}.pkl")

if __name__ == "__main__":
    for i, config in enumerate(hyperparams_list):
        for j in range(5):  # Repeat 5 times
            run_id = f"{chr(65+i)}_{j+1}"  # A_1, A_2, ..., E_5
            run_experiment(run_id, config)