import os
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("SmartHome_RL")
runs = client.search_runs(experiment.experiment_id)

data = []

for run in runs:
    run_id = run.info.run_id
    params = run.data.params
    metrics = run.data.metrics
    final_reward = metrics.get("reward")
    data.append({
        "run_name": run.data.tags.get("mlflow.runName"),
        "learning_rate": float(params.get("lr", 0)),
        "gamma": float(params.get("gamma", 0)),
        "epsilon": float(params.get("epsilon", 0)),
        "final_reward": final_reward
    })

df = pd.DataFrame(data)
df = df.sort_values("final_reward", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(df["run_name"], df["final_reward"], color="skyblue")
plt.xticks(rotation=45)
plt.title("Final Reward per Run")
plt.ylabel("Reward (Higher is Better)")
plt.tight_layout()
plt.show()
