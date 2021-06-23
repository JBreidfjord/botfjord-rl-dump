# %%
import h5py
import numpy as np
from dlchess.agents.zero import ZeroAgent
from dlchess.encoders.zero import ZeroEncoder
from dlchess.rl.experience import load_experience
from tensorflow.keras.models import load_model

weights = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 10]
num_exp_files = 16

file_indices = np.random.permutation(range(num_exp_files))
results = {}
# %%
for policy_weight in weights:
    for value_weight in weights:
        model = load_model("dlchess/models/zero_0.h5", compile=False)
        encoder = ZeroEncoder()
        agent = ZeroAgent(model, encoder)

        loss_weights = (policy_weight, value_weight)
        print(loss_weights)

        histories = [None] * num_exp_files
        for i in file_indices:
            experience = load_experience(
                h5py.File(f"data/zero_exp_0.{i}.h5", mode="r"), zero=True
            )
            history = agent.train(
                experience,
                epochs=1,
                learning_rate=0.01,
                batch_size=1024,
                loss_weights=list(loss_weights),
            )
            histories[i] = history.history

        results[loss_weights] = histories
        print("")
        print(histories)
        print("")
# %%
final_losses = {w: l[file_indices[-1]] for w, l in results.items()}

best_losses = {}
for weights, histories in results.items():
    lows = (
        min([x["loss"][0] for x in histories]),
        min([x["policy_output_loss"][0] for x in histories]),
        min([x["value_output_loss"][0] for x in histories]),
    )
    best_losses[weights] = lows

best_policy_weight = None
best_value_weight = None
best_total_weight = None
best_policy_loss = np.inf
best_value_loss = np.inf
best_total_loss = np.inf  # policy + value, not the calculated loss value
for weights, histories in results.items():
    for x in histories:
        p = x["policy_output_loss"][0]
        v = x["value_output_loss"][0]
        t = p + v
        if p < best_policy_loss:
            best_policy_loss = p
            best_policy_weight = weights
        if v < best_value_loss:
            best_value_loss = v
            best_value_weight = weights
        if t < best_total_loss:
            best_total_loss = t
            best_total_weight = weights
# %%
print("\nFinal Losses")
for k, v in final_losses.items():
    print(k, v, end="\n\n")
# %%
print("\nBest Losses")
for k, v in best_losses.items():
    print(k, v, end="\n\n")
print("")
# %%
print("Best Policy Weights", best_policy_weight, best_policy_loss)
print("Best Value Weights", best_value_weight, best_value_loss)
print("Best Total Weights", best_total_weight, best_total_loss)
# %%
for i in range(num_exp_files):
    print(f"\nStep {i}")
    result = [(w, l[i]) for w, l in results.items()]
    result.sort(
        key=lambda x: x[1]["policy_output_loss"][0] + x[1]["value_output_loss"][0]
    )
    for r in result:
        print(r[0], r[1])

# %%
