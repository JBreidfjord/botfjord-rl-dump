import logging
import os
from random import shuffle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tqdm.auto import tqdm

from dlchess.agents.prime import PrimeAgent
from dlchess.rl.experience import load_experience

model_name = "omega"
size = "large"

logging.basicConfig(
    filename=f"logs/{model_name}_{size}_sl.log", encoding="utf-8", level=logging.INFO
)

plot = False
learning_rate = 0.01
batch_size = 2048
epochs = 10
validation_split = 0.05
loss_weights = [1, 1]

logging.info(
    f"LR {learning_rate} | Batch Size {batch_size} | Epochs {epochs} | Val Split {validation_split} | Loss Weights {loss_weights}"
)

model = load_model(f"dlchess/models/{model_name}_{size}_0")
agent = PrimeAgent(model, None)
opt = RMSprop(learning_rate=learning_rate)
agent.opt = opt
agent.model.optimizer = None

exp_files = [
    f.path
    for f in os.scandir(f"/data/code/botfjord/sl/{model_name}/")
    if not f.name.startswith("progress")
    and not f.name.startswith("temp")
    and not f.name.endswith(".used")
]
shuffle(exp_files)

i = 0
while i < len(exp_files):
    exp = load_experience(
        h5py.File(f"/data/code/botfjord/sl/{model_name}/progress_partial.h5", mode="r"),
        zero=True,
    )
    for f in tqdm(exp_files[i : i + 20], desc="Merging Experience"):
        new_exp = load_experience(h5py.File(f, mode="r"), zero=True)
        if np.isnan(new_exp.rewards).any():
            mask = ~np.isnan(new_exp.rewards)
            new_exp.states = new_exp.states[mask]
            new_exp.visit_counts = new_exp.visit_counts[mask]
            new_exp.rewards = new_exp.rewards[mask]
            new_exp.serialize(h5py.File(f, mode="w"))
            print("Removed NaN")
        exp.merge_with(new_exp)

    skip = len(
        [
            f
            for f in os.listdir(f"/data/code/botfjord/sl/{model_name}/")
            if f.startswith("progress") and not f.endswith("partial.h5")
        ]
    )
    exp.shuffle()
    exp.serialize_batches(
        100_000, f"/data/code/botfjord/sl/{model_name}/progress", even=False, skip=skip
    )

    for f in tqdm(exp_files[i : i + 20], desc="Renaming Files"):
        os.rename(f, f"{f}.used")

    i += 20

merged_files = [
    f.path
    for f in os.scandir(f"/data/code/botfjord/sl/{model_name}/")
    if f.name.startswith("progress") and not f.name.endswith("partial.h5")
]
shuffle(merged_files)

i = 0
for f in tqdm(merged_files, desc="Training steps"):
    exp = load_experience(h5py.File(f, mode="r"), zero=True)

    if np.isnan(exp.rewards).any():
        mask = ~np.isnan(exp.rewards)
        exp.states = exp.states[mask]
        exp.visit_counts = exp.visit_counts[mask]
        exp.rewards = exp.rewards[mask]
        exp.serialize(h5py.File(f, mode="w"))
        print("Removed NaN")

    exp.shuffle()

    if i % 3 == 0 and i > 0:
        opt = RMSprop(learning_rate=learning_rate / 10)
        agent.opt = opt
        agent.model.optimizer = None

    if i % 30 == 0 and i > 0:
        opt = RMSprop(learning_rate=learning_rate)
        agent.opt = opt
        agent.model.optimizer = None
    # elif i == 10:
    #     opt = RMSprop(learning_rate=learning_rate * 100)
    #     agent.opt = opt
    #     agent.model.optimizer = None
    # elif i == 15:
    #     opt = RMSprop(learning_rate=learning_rate)
    #     agent.opt = opt
    #     agent.model.optimizer = None

    history = agent.train(
        exp,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        loss_weights=loss_weights,
    )

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    p_loss = history.history["policy_output_loss"]
    val_p_loss = history.history["val_policy_output_loss"]
    v_loss = history.history["value_output_loss"]
    val_v_loss = history.history["val_value_output_loss"]

    ep = range(1, len(loss) + 1)

    if plot:
        plt.plot(ep, loss, "bo", label="Training loss")
        plt.plot(ep, val_loss, "b", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.figure()

        plt.plot(ep, p_loss, "bo", label="Training policy loss")
        plt.plot(ep, val_p_loss, "b", label="Validation policy loss")
        plt.title("Training and Validation Policy Loss")
        plt.legend()

        plt.figure()

        plt.plot(ep, v_loss, "bo", label="Training value loss")
        plt.plot(ep, val_v_loss, "b", label="Validation value loss")
        plt.title("Training and Validation Value Loss")
        plt.legend()

        plt.show()

    logging.info(
        f"Average Validation Loss {round(np.average(val_loss), 3)} | Training Loss {round(np.average(loss), 3)}"
    )
    logging.info(
        f"Average Validation Policy Loss {round(np.average(val_p_loss), 3)} | Training Policy Loss {round(np.average(p_loss), 3)}"
    )
    logging.info(
        f"Average Validation Value Loss {round(np.average(val_v_loss), 3)} | Training Value Loss {round(np.average(v_loss), 3)}\n"
    )

    i += 1


agent.serialize(f"dlchess/models/{model_name}_{size}_1")
agent.serialize(f"dlchess/models/{model_name}_{size}_progress")
