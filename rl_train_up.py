import os
import random

import h5py
import tensorflow as tf
from dlchess.agents.zero import ZeroAgent
from dlchess.encoders.ac12 import AC12Encoder
from dlchess.encoders.twelveplane import TwelvePlaneEncoder
from dlchess.encoders.zero import ZeroEncoder
from dlchess.rl.ac import ActorCriticAgent
from dlchess.rl.experience import ZeroCollector, combine_experience, load_experience
from dlchess.rl.q import QAgent
from tensorflow.keras.models import load_model
from tqdm.auto import tqdm

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = load_model("dlchess/models/prime_5")
encoder = ZeroEncoder()
agent = ZeroAgent(model, encoder)

# For data loading
model_name = "zero"
model_version = str(0)

exp_files = [
    f.path for f in os.scandir(f"/data/code/botfjord/data/{model_name}_{model_version}")
]
lichess_used_files = [
    f.path for f in os.scandir("/data/code/botfjord/data/lichess/used") if f.is_file()
]
lichess_files = [
    f.path for f in os.scandir("/data/code/botfjord/data/lichess") if f.is_file()
]
files = exp_files # + lichess_files + lichess_used_files
random.shuffle(files)

i = 0
while i < len(files):
    break
    # remove obv
    collector = ZeroCollector()
    while len(collector.rewards) < 50000 and i < len(files):
        print(i)
        try:
            experience = load_experience(h5py.File(files[i], mode="r"), zero=True)
            experience.remove_draws()
            experience = experience.to_collector()
            experience.shuffle()
        except KeyError:
            i += 1
            continue

        # Prevent error for trying to concat empty arrays
        if collector.rewards or experience.rewards:
            collector = combine_experience(
                [collector, experience], zero=True
            ).to_collector()
        print(len(collector.rewards))
        print("")
        i += 1

    collector.shuffle()
    collector.to_buffer().serialize(
        h5py.File(f"/data/code/botfjord/no_draw_concat/2/{i}.h5", mode="w")
    )
    print("Buffer saved")

# buffers = [
#     f.path for f in os.scandir("/data/code/botfjord/no_draw_concat/2/") if f.is_file()
# ]
# random.shuffle(buffers)
# for buffer in tqdm(buffers, position=1):
#     experience = load_experience(h5py.File(buffer, mode="r"), zero=True)
#     agent.train(experience, batch_size=1024, verbose=1, ignore_draws=True)

# agent.serialize(f"dlchess/models/prime_5")
