from __future__ import annotations

import logging
import multiprocessing as mp
import os
import timeit

import h5py
import tensorflow as tf
from dlchess.agents.prime import PrimeAgent
from dlchess.encoders.theta import ThetaEncoder
from dlchess.rl.experience import (
    ExperienceBuffer,
    ZeroCollector,
    combine_experience,
    load_experience,
)
from log_to_pgn import pgn_generator
from tensorflow.keras import models
from tensorflow.keras.optimizers import RMSprop
from tqdm import tqdm, trange

import chess
import chess.engine

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def worker(
    game_queue,
    queue,
    encoder,
    temperature,
    model_name,
    model_version,
    search_rounds,
    _id,
):
    logging.basicConfig(
        filename=f"logs/{model_name}_training.log", encoding="utf-8", level=logging.INFO
    )

    # if get_exp_step(model_name, model_version) == 0:
    model = models.load_model(
        f"dlchess/models/{model_name}_{model_version}", compile=False
    )
    white_player = PrimeAgent(model, encoder, num_rounds=search_rounds)
    black_player = PrimeAgent(model, encoder, num_rounds=search_rounds)
    # else:
    #     model = models.load_model(
    #         f"dlchess/models/{model_name}_{model_version}", compile=False
    #     )
    #     progress_model = models.load_model(f"dlchess/models/{model_name}_progress")
    #     if np.random.randint(0, 2):
    #         white_player = PrimeAgent(progress_model, encoder, num_rounds=search_rounds)
    #         black_player = PrimeAgent(model, encoder, num_rounds=search_rounds)
    #     else:
    #         white_player = PrimeAgent(model, encoder, num_rounds=search_rounds)
    #         black_player = PrimeAgent(progress_model, encoder, num_rounds=search_rounds)

    white_collector = ZeroCollector()
    black_collector = ZeroCollector()
    white_player.set_collector(white_collector)
    black_player.set_collector(black_collector)
    white_player.set_temperature(temperature)
    black_player.set_temperature(temperature)

    # white_player.prevent_repetition = True
    # black_player.prevent_repetition = True

    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    w = 0
    d = 0
    b = 0
    p = 0
    total_white = 0
    total_black = 0
    game_id = game_queue.get()
    while game_id is not False:
        start_time = timeit.default_timer()
        white_collector.begin_episode()
        black_collector.begin_episode()

        # try:
        game_record, game = simulate_game(white_player, black_player)
        # except:
        #     continue

        if game_record is None:
            game_score = engine.analyse(game, limit=chess.engine.Limit(depth=15))[
                "score"
            ]
            # white_reward = np.tanh(
            #     np.tanh(np.tanh(((game_score.white().score())) / 1000) * 0.75) * 7.5
            # )
            white_reward = round(
                ((game_score.white().wdl().expectation() - 0.5) / 0.5) / 2, 3
            )
            black_reward = round(
                ((game_score.black().wdl().expectation() - 0.5) / 0.5) / 2, 3
            )
            assert -1 <= white_reward <= 1 and -1 <= black_reward <= 1
            # print(game_score)
            d += 1
            winner = "D"
        elif game_record:
            white_reward = 1.0
            black_reward = -1.0
            w += 1
            winner = "W"
            move_stack = []
            board = chess.Board()
            for mv in game.move_stack:
                move_stack.append(board.san_and_push(mv))
            logging.info("White win")
            logging.info(pgn_generator(move_stack))
        else:
            white_reward = -1.0
            black_reward = 1.0
            b += 1
            winner = "B"
            move_stack = []
            board = chess.Board()
            for mv in game.move_stack:
                move_stack.append(board.san_and_push(mv))
            logging.info("Black win")
            logging.info(pgn_generator(move_stack))

        white_collector.complete_episode(white_reward)
        black_collector.complete_episode(black_reward)
        total_white += white_reward
        total_black += black_reward

        p += 1
        game_time = round(timeit.default_timer() - start_time)

        output_1 = f"Worker {_id} Game {game_id}".ljust(16) + " | "
        output_2 = f"{winner} ({game.fullmove_number})".ljust(7) + " | "
        output_3 = (
            f"WR {round(white_reward, 1)} BR {round(black_reward, 1)}".ljust(14) + " | "
        )
        output_4 = f"W {int((w/p)*100)}% ({w})".ljust(11) + " | "
        output_5 = f"D {int((d/p)*100)}% ({d})".ljust(11) + " | "
        output_6 = f"B {int((b/p)*100)}% ({b})".ljust(11) + " | "
        output_7 = f"{game_time}s ({round(game_time/game.fullmove_number, 1)}s/mv)"
        print(
            output_1 + output_2 + output_3 + output_4 + output_5 + output_6 + output_7
        )

        game_id = game_queue.get()

    engine.quit()
    experience = combine_experience([white_collector, black_collector], zero=True)
    queue.put((experience, _id))
    # print(f"Process {_id} reached termination")


def mp_simulate(
    num_games, num_procs, encoder, temperature, model_name, model_version, search_rounds
):
    # Start processes
    queue = mp.Queue()
    game_queue = mp.Queue(maxsize=num_games + num_procs)
    task_list = list(range(num_games)) + [False] * num_procs
    for task in task_list:
        game_queue.put(task)

    procs: dict[int, mp.Process] = {}
    for i in range(num_procs):
        p = mp.Process(
            target=worker,
            args=(
                game_queue,
                queue,
                encoder,
                temperature,
                model_name,
                model_version,
                search_rounds,
                i,
            ),
        )
        procs[i] = p
        p.start()
        print(f"Started sim process {i}")

    # Grab data from processes
    for i in range(num_procs):
        new_experience, proc_id = queue.get()
        # print(f"Grabbed experience from {proc_id}")
        if num_procs > 1:
            if i == 0:
                experience = new_experience.to_collector()
            else:
                experience = combine_experience(
                    [experience, new_experience.to_collector()], zero=True
                )
        else:
            experience = new_experience

        # print(f"Attempting to join process {proc_id}")
        p = procs[proc_id]
        p.join()
        print(f"Finished sim process {proc_id}")

    return experience


def simulate_game(white_player, black_player, verbose=0):
    game = chess.Board()
    agents = {True: white_player, False: black_player}

    while not game.is_game_over(claim_draw=True):
        if verbose:
            print(game)
            print("white to move" if game.turn else "black to move")
        next_move = agents[game.turn].select_move(game)
        game.push(next_move)
        if verbose:
            print(next_move, end="\n\n")
        if game.fullmove_number > 50:
            return None, game

    game_result = game.outcome(claim_draw=True)

    # if __name__ == "__main__":
    #     print(game)
    #     print(game_result)
    #     print("")

    return game_result.winner, game


def evaluate_model(agent1, agent2, num_games, target, verbose=0):
    wins = 0
    losses = 0
    draws = 0
    played = 0
    color = True

    agent1.set_collector(None)
    agent2.set_collector(None)

    t = trange(num_games)
    for _ in t:
        if color:
            white_player, black_player = agent1, agent2
        else:
            black_player, white_player = agent1, agent2

        game_record, _ = simulate_game(white_player, black_player, verbose=verbose)
        if game_record is None:
            draws += 1
        elif game_record == color:
            wins += 1
        else:
            losses += 1
        played += 1
        t.set_description(
            f"W {int((wins / played) * 100)}% ({wins}) | D {int((draws / played) * 100)}% ({draws}) | L {int((losses / played) * 100)}% ({losses})"
        )

        color = not color

    print(f"Agent record: W{wins} D{draws} L{losses}")
    win_rate = round((wins + (draws / 2)) / played, 2)
    print(f"Win Rate: {round(win_rate * 100, 1)}%\n")

    return win_rate >= target


def get_exp_step(model_name, model_version):
    step = None
    for f in os.listdir(f"/data/code/botfjord/data/{model_name}_{model_version}"):
        if f.endswith(".h5"):
            if step is None:
                step = 0
            new_step = int(f.removeprefix("exp_").removesuffix(".h5"))
            step = new_step if new_step > step else step

    return step + 1 if step is not None else 0


def check_lichess(experience_buffer: ExperienceBuffer):
    """Checks for stored game data for games played on Lichess"""
    lichess_data_dir = "/data/code/botfjord/data/lichess"
    lichess_games = [f for f in os.listdir(lichess_data_dir) if f.endswith(".h5")][:5]
    # Limit to 5 to avoid OOM and to spread training
    collectors = [experience_buffer.to_collector()]

    if not lichess_games:
        return experience_buffer

    for game_file in tqdm(lichess_games, desc="Lichess Games"):
        try:
            exp = load_experience(
                h5py.File(os.path.join(lichess_data_dir, game_file), mode="r"),
                zero=True,
            ).to_collector()
        except:
            print(f"Error loading {game_file}")
            continue
        collectors.append(exp)
        os.rename(
            os.path.join(lichess_data_dir, game_file),
            os.path.join(lichess_data_dir, "used", game_file),
        )

    return combine_experience(collectors, zero=True)


def manager(model_name, model_version, exp_step, result_queue, encoder):
    processes = 8

    num_games = 200
    eval_games = 10
    eval_search_rounds = 200
    eval_target = 0.60

    temperature = 5
    search_rounds = 50
    batch_size = 1024

    if num_games < processes:
        num_games = processes
        print(
            "WARN: Number of games must be equal to or greater than number of processes"
        )

    model = models.load_model(
        f"dlchess/models/{model_name}_{model_version}", compile=(model_version > 0)
    )

    experience = mp_simulate(
        num_games,
        processes,
        encoder,
        temperature,
        model_name,
        model_version,
        search_rounds,
    )
    experience.serialize(
        h5py.File(
            f"/data/code/botfjord/data/{model_name}_{model_version}/exp_{exp_step}.h5",
            mode="w",
        )
    )

    if exp_step > 0:
        in_progress_model = models.load_model(f"dlchess/models/{model_name}_progress")
        white_player = PrimeAgent(in_progress_model, encoder, eval_search_rounds)
    else:
        white_player = PrimeAgent(model, encoder, eval_search_rounds)
    black_player = PrimeAgent(model, encoder, eval_search_rounds)

    exp = load_experience(
        h5py.File(
            f"/data/code/botfjord/data/{model_name}_{model_version}/exp_{exp_step}.h5",
            mode="r",
        ),
        zero=True,
    )
    exp = check_lichess(exp)

    exp.shuffle()

    white_player.train(exp, batch_size)
    white_player.serialize(f"dlchess/models/{model_name}_progress")

    if exp_step % 10 == 0 and exp_step > 0:
        if evaluate_model(white_player, black_player, eval_games, eval_target):
            white_player.serialize(f"dlchess/models/{model_name}_{model_version + 1}")
            result_queue.put(True)
        else:
            result_queue.put(False)
    else:
        result_queue.put(False)


if __name__ == "__main__":
    model_name = "theta_small"
    model_version = 1
    target_version = 10

    logging.basicConfig(
        filename=f"logs/{model_name}_training.log", encoding="utf-8", level=logging.INFO
    )

    os.makedirs(f"/data/code/botfjord/data/{model_name}_{model_version}", exist_ok=True)

    encoder = ThetaEncoder()

    mp.set_start_method("spawn")

    while model_version < target_version:
        exp_step = get_exp_step(model_name, model_version)
        print(exp_step)
        result_queue = mp.Queue()
        mng = mp.Process(
            target=manager,
            args=(model_name, model_version, exp_step, result_queue, encoder),
        )
        mng.start()
        print("Started manager process")
        result = result_queue.get()
        mng.join()
        print("Finished manager process\n")
        if result:
            model_version += 1
            print("Model passed evaluation")
            print(f"New model: {model_name}_{model_version}")
