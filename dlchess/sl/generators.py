import logging
import multiprocessing as mp
import os
import time
from secrets import token_hex

import h5py
import numpy as np
import tensorflow as tf
from dlchess.agents.base import Agent
from dlchess.agents.prime import PrimeAgent
from dlchess.encoders.base import Encoder
from dlchess.encoders.prime import PrimeEncoder
from dlchess.rl.experience import ExperienceCollector, ZeroCollector, combine_experience
from tensorflow.keras.models import Model, load_model
from tqdm.auto import tqdm

import chess
import chess.engine
import chess.pgn

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

logging.basicConfig(filename="logs/primer.log", encoding="utf-8", level=logging.INFO)


def collect(queue: mp.Queue, task_queue: mp.Queue):
    try:
        model = load_model(f"dlchess/models/prime_progress")
    except:
        model = load_model(f"dlchess/models/prime_0", compile=False)

    encoder = PrimeEncoder()
    collector = ZeroCollector()
    agent = PrimeAgent(model, encoder, num_rounds=100, collector=collector)
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    position_fen = task_queue.get()
    while position_fen is not None:
        game_state = chess.Board(position_fen)
        color = game_state.turn

        analysis = engine.analyse(
            game_state,
            limit=chess.engine.Limit(depth=10),
            info=chess.engine.INFO_ALL,
            game=object(),
            multipv=game_state.legal_moves.count(),
        )

        if analysis is None:
            engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
            collector.reset_episode()
            continue

        engine_result = (
            analysis[0]["score"].pov(color).score(mate_score=10000),
            analysis[0]["pv"][0],
        )

        collector.begin_episode()
        move = agent.select_move(game_state)

        if move == engine_result[1]:
            cp_loss = 0
            agent_result = engine_result[0]
        else:
            for pv in analysis:
                if move == pv["pv"][0]:
                    agent_result = pv["score"].pov(color).score(mate_score=10000)
            cp_loss = engine_result[0] - agent_result

        reward = max((100 - cp_loss) / 100, -1)

        reward_spec = np.linspace(-1, 1, len(analysis))[::-1]
        for i, res in enumerate(analysis):
            if res["pv"][0] == move:
                reward = reward_spec[i]

        collector.complete_episode(reward)

        position_fen = task_queue.get()
        remaining_tasks = task_queue.qsize()
        if remaining_tasks % 100 == 0:
            print(f"{remaining_tasks} tasks remaining")

    queue.put(collector)


class Primer:
    def __init__(
        self,
        data_dir: str,
        model_name: str,
        game_batch_size: int = 100,
        workers: int = 4,
        context=None,
        queue=None,
        task_queue=None,
    ):
        self.data_dir = data_dir
        self.file_nums = None
        self.batch_size = game_batch_size

        self.model_name = model_name
        self.workers = workers
        self.context = context
        self.queue = queue
        self.task_queue = task_queue

        self.games = self.game_generator()

    def manage_collection(self):
        collectors = []

        position_fens = self.position_generator()
        position_fens.extend([None] * self.workers)
        for position_fen in position_fens:
            self.task_queue.put(position_fen)

        procs = []
        for i in range(self.workers):
            p = mp.Process(target=collect, args=(self.queue, self.task_queue))
            p.start()
            print(f"Started worker process {i}")
            procs.append(p)

        for i, p in enumerate(procs):
            collector = self.queue.get()
            if collector is not None:
                collectors.append(collector)
                print(f"Received collector from process {i}")
            else:
                print(f"Received none from process {i}")

        for p in procs:
            p.terminate()
            p.join()
            print(f"Finished worker process {i}")

        if collectors == []:
            return

        experience = combine_experience(collectors, zero=True)

        reward_avg = np.average(experience.rewards)
        print(f"Average reward: {reward_avg}")
        print(f"Average accuracy: {round(((reward_avg * 0.5) + 0.5) * 100, 1)}%")
        logging.info(f"Average reward: {reward_avg}")
        logging.info(f"Average accuracy: {round(((reward_avg * 0.5) + 0.5) * 100, 1)}%")

        exp_filename = f"{token_hex(8)}.h5"
        experience.serialize(
            h5py.File(
                f"/data/code/botfjord/data/{self.model_name}/{exp_filename}", mode="w"
            )
        )
        return exp_filename

    def generate(
        self, exp_queue: mp.Queue, num_rounds: int = None, file_nums: list[int] = None
    ):
        if file_nums is not None:
            self.file_nums = file_nums
            logging.info(f"Set file_nums to {self.file_nums}")

        _active = True
        i = 0
        while _active:
            exp_filename = self.manage_collection()
            if exp_filename is None:
                _active = False
                continue

            exp_queue.put(exp_filename)
            # logging.info("Experience placed in queue")
            # time.sleep(60)
            # Allows training to occur so next iteration can use updated model

            i += 1
            if num_rounds is not None:
                if i == num_rounds:
                    _active = False

    def game_generator(self):
        pgn_files = [f for f in os.listdir(self.data_dir) if f.endswith(".pgn")]
        if self.file_nums is not None:
            x = []
            for i in self.file_nums:
                x.append(pgn_files[i])
            pgn_files = x

        for pgn_file in pgn_files:
            logging.info(f"Using PGN file {pgn_file}")
            with open(os.path.join(self.data_dir, pgn_file), mode="r") as pgn:
                _active = True
                while _active:
                    pgn_game = chess.pgn.read_game(pgn)
                    if pgn_game is not None:
                        yield pgn_game
                    else:
                        _active = False

    def game_batcher(self):
        i = 0
        game_batch = []
        for game in self.games:
            i += 1
            game_batch.append(game)
            if i == self.batch_size:
                return game_batch

    def position_generator(self):
        raw_games = self.game_batcher()
        position_fens = []
        for game in raw_games:
            board = game.board()
            position_fens.append(board.fen())
            for move in game.mainline_moves():
                board.push(move)
                if not board.is_game_over():
                    position_fens.append(board.fen())

        return position_fens
