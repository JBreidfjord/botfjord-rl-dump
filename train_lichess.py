import os
import random

import h5py
import tensorflow as tf
from dlchess.agents.prime import PrimeAgent
from dlchess.encoders.ac12 import AC12Encoder
from dlchess.encoders.prime import PrimeEncoder
from dlchess.encoders.twelveplane import TwelvePlaneEncoder
from dlchess.rl.ac import ActorCriticAgent
from dlchess.rl.experience import ZeroCollector, combine_experience, load_experience
from dlchess.rl.q import QAgent
from tensorflow.keras.models import load_model
from tqdm.auto import tqdm

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = load_model("dlchess/models/prime_progress")
encoder = PrimeEncoder()
agent = PrimeAgent(model, encoder)

lichess_files = [
    f.path for f in os.scandir("/data/code/botfjord/data/lichess") if f.is_file()
]
random.shuffle(lichess_files)

for file in lichess_files:
    experience = load_experience(h5py.File(file, mode="r"), zero=True)
    agent.train(experience, batch_size=256, verbose=1)

agent.serialize(f"dlchess/models/prime_progress")
