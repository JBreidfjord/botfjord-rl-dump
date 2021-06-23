import os
from random import shuffle

import h5py
from dlchess.agents.prime import PrimeAgent
from dlchess.encoders.theta import ThetaEncoder
from dlchess.rl.experience import load_experience
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

model = load_model("dlchess/models/theta_small_progress")
encoder = ThetaEncoder()
agent = PrimeAgent(model, encoder)

opt = RMSprop(learning_rate=0.2)
agent.opt = opt
agent.model.optimizer = None

# For data loading
model_name = "theta_small_1"

num_epochs = 1
exp_files = [f.path for f in os.scandir(f"/data/code/botfjord/data/{model_name}")]
shuffle(exp_files)

exp = load_experience(h5py.File(exp_files[0], mode="r"), zero=True)
for f in exp_files[1:]:
    new_exp = load_experience(h5py.File(f, mode="r"), zero=True)
    exp.merge_with(new_exp)
    exp.shuffle()

exp.serialize(h5py.File("/data/code/botfjord/data/theta_small_1/exp_0.h5", mode="w"))
# agent.train(exp, epochs=10, validation_split=0.2)
# agent.serialize("dlchess/models/theta_small_progress_")
