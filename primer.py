import logging
import multiprocessing as mp
import time

import h5py
import tensorflow as tf
from dlchess.agents.prime import PrimeAgent
from dlchess.encoders.prime import PrimeEncoder
from dlchess.rl.experience import load_experience
from dlchess.sl.generators import Primer
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop


def trainer(exp_filename: str, model_name: str = "prime", round_num: int = 0):
    try:
        model = load_model(f"dlchess/models/{model_name}_progress")
    except:
        model = load_model(f"dlchess/models/{model_name}_0")
    exp = load_experience(
        h5py.File(f"/data/code/botfjord/data/{model_name}/{exp_filename}", mode="r"),
        zero=True,
    )
    agent = PrimeAgent(model, PrimeEncoder())

    opt = RMSprop(learning_rate=0.0002)
    if round_num % 100 == 0 and round_num > 0:
        agent.opt = opt
        agent.model.optimizer = None

    # temp
    if round_num == 10:
        opt = RMSprop(learning_rate=0.002)
        agent.opt = opt
        agent.model.optimizer = None

    exp.shuffle()
    agent.train(exp, batch_size=64)
    agent.serialize(f"dlchess/models/{model_name}_progress")

    if round_num % 10 == 0 and round_num > 0:
        agent.serialize(f"dlchess/models/{model_name}_cp_{int(round_num / 10)}")


def manager(num_rounds, model_name, exp_queue, queue, task_queue, context):
    primer = Primer(
        "/home/jbreid/code/deeplearn/chess/data/fics",
        model_name=model_name,
        game_batch_size=100,
        workers=10,
        context=context,
        queue=queue,
        task_queue=task_queue,
    )

    for i in range(num_rounds):
        logging.info(f"Started iteration {i}")
        primer.generate(exp_queue, num_rounds=1, file_nums=[4, 5])
        time.sleep(60)
        logging.info(f"Finished iteration {i}")


if __name__ == "__main__":
    #    physical_devices = tf.config.list_physical_devices("GPU")
    #   tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logging.basicConfig(
        filename="logs/primer.log", encoding="utf-8", level=logging.INFO
    )

    mp.set_start_method("spawn")
    context = mp.get_context("spawn")

    sync_manager = mp.Manager()
    queue = sync_manager.Queue()
    exp_queue = sync_manager.Queue()
    task_queue = sync_manager.Queue()

    model_name = "prime"
    num_rounds = 200

    mng = mp.Process(
        target=manager,
        args=(num_rounds, model_name, exp_queue, queue, task_queue, context),
    )
    mng.start()
    print("Started manager process")

    for i in range(num_rounds):
        exp_filename = exp_queue.get()
        # logging.info("Experience received from queue")
        train = mp.Process(target=trainer, args=(exp_filename, model_name, i))
        train.start()
        print(f"Started training process for iteration {i}")
        train.join()
        print(f"Finished training process for iteration {i}\n")

    mng.join()
    print("Finished manager process")
