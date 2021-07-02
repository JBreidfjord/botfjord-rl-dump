from __future__ import annotations

import logging
import multiprocessing as mp
import os
import timeit

import chess
import chess.engine
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD, RMSprop
from tqdm import tqdm, trange

from dlchess.agents.prime import PrimeAgent
from dlchess.encoders.omega import OmegaEncoder
from dlchess.encoders.theta import ThetaEncoder
from dlchess.encoders.zero import ZeroEncoder
from dlchess.rl.experience import (
    ExperienceBuffer,
    ZeroCollector,
    combine_experience,
    load_experience,
)
from log_to_pgn import pgn_generator

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

    """Revert this back"""
    # model = models.load_model(
    #     f"dlchess/models/{model_name}_{model_version}", compile=False
    # )

    # Gives 1/10 chance to use an older model version
    # Idea is to add variety to opponents, prevent memorization,
    # and prevent forgetting how to play weaker opponents
    white_models = [
        f.path
        for f in os.scandir(f"dlchess/models/")
        if f.name.startswith(f"{model_name}_white_") and not f.name.endswith("progress")
    ]
    black_models = [
        f.path
        for f in os.scandir(f"dlchess/models/")
        if f.name.startswith(f"{model_name}_black_") and not f.name.endswith("progress")
    ]
    if white_models:
        w_probs = [0.1 / len(white_models)] * len(white_models)
        white_models.append(f"dlchess/models/{model_name}_white_progress")
        w_probs.append(0.9)
        white_path = np.random.choice(white_models, p=w_probs)
    else:
        white_path = f"dlchess/models/{model_name}_white_progress"
    if black_models:
        b_probs = [0.1 / len(black_models)] * len(black_models)
        black_models.append(f"dlchess/models/{model_name}_black_progress")
        b_probs.append(0.9)
        black_path = np.random.choice(black_models, p=b_probs)
    else:
        black_path = f"dlchess/models/{model_name}_black_progress"

    white_model = models.load_model(white_path, compile=False)
    black_model = models.load_model(black_path, compile=False)

    white_player = PrimeAgent(white_model, encoder, num_rounds=search_rounds)
    black_player = PrimeAgent(black_model, encoder, num_rounds=search_rounds)

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

        game_record, game = simulate_game(white_player, black_player)

        if game_record is None:
            # game_score = engine.analyse(game, limit=chess.engine.Limit(depth=15))[
            #     "score"
            # ]
            # white_reward = round(
            #     ((game_score.white().wdl().expectation() - 0.5) / 0.5) / 2, 3
            # )
            # black_reward = round(
            #     ((game_score.black().wdl().expectation() - 0.5) / 0.5) / 2, 3
            # )
            # assert -1 <= white_reward <= 1 and -1 <= black_reward <= 1
            d += 1
            winner = "D"
        elif game_record:
            # white_reward = 1.0
            # black_reward = -1.0
            w += 1
            winner = "W"
            move_stack = []
            board = chess.Board()
            for mv in game.move_stack:
                move_stack.append(board.san_and_push(mv))
            logging.info(f"White win")
            logging.info(pgn_generator(move_stack))
        else:
            # white_reward = -1.0
            # black_reward = 1.0
            b += 1
            winner = "B"
            move_stack = []
            board = chess.Board()
            for mv in game.move_stack:
                move_stack.append(board.san_and_push(mv))
            logging.info(f"Black win")
            logging.info(pgn_generator(move_stack))

        if len(game.move_stack) % 2 == 0:
            num_white_moves = int(len(game.move_stack) / 2)
            num_black_moves = int(len(game.move_stack) / 2)
        else:
            num_white_moves = int((len(game.move_stack) + 1) / 2)
            num_black_moves = int((len(game.move_stack) - 1) / 2)

        white_reward = np.mean(white_player.collector.rewards[-num_white_moves:])
        black_reward = np.mean(black_player.collector.rewards[-num_black_moves:])
        white_collector.complete_episode(None)
        black_collector.complete_episode(None)
        total_white += white_reward
        total_black += black_reward

        # w_rewards = np.array(white_player.collector.rewards)
        # b_rewards = np.array(black_player.collector.rewards)
        # if winner == "W":
        #     w_rewards[-num_white_moves:] = 1
        #     b_rewards[-num_black_moves:] = -1
        # elif winner == "B":
        #     w_rewards[-num_white_moves:] = -1
        #     b_rewards[-num_black_moves:] = 1
        # white_player.collector.rewards = w_rewards.tolist()
        # black_player.collector.rewards = b_rewards.tolist()

        # white_reward = np.mean(white_player.collector.rewards[-num_white_moves:])
        # black_reward = np.mean(black_player.collector.rewards[-num_black_moves:])

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
        output_7 = f"{game_time}s ({round(game_time/game.ply(), 1)}s/ply)"
        print(
            output_1 + output_2 + output_3 + output_4 + output_5 + output_6 + output_7
        )

        game_id = game_queue.get()

    white_avg_loss = round((1 - (total_white / p)) * 100)
    black_avg_loss = round((1 - (total_black / p)) * 100)

    engine.quit()
    # experience = combine_experience([white_collector, black_collector], zero=True)
    white_exp = white_collector.to_buffer()
    black_exp = black_collector.to_buffer()
    queue.put((white_exp, black_exp, _id, white_avg_loss, black_avg_loss))
    # print(f"Process {_id} reached termination")


def mp_simulate(
    num_games, num_procs, encoder, temperature, model_name, model_version, search_rounds
):
    logging.basicConfig(
        filename=f"logs/{model_name}_training.log", encoding="utf-8", level=logging.INFO
    )
    context = mp.get_context("spawn")
    # Start processes
    queue = context.Queue()
    game_queue = context.Queue(maxsize=num_games + num_procs)
    task_list = list(range(num_games)) + [False] * num_procs
    for task in task_list:
        game_queue.put(task)

    procs: dict[int, mp.Process] = {}
    for i in range(num_procs):
        p = context.Process(
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
    white_loss_list = []
    black_loss_list = []
    for i in range(num_procs):
        new_white_exp, new_black_exp, proc_id, white_loss, black_loss = queue.get()
        # print(f"Grabbed experience from {proc_id}")
        if num_procs > 1:
            if i == 0:
                white_exp = new_white_exp.to_collector()
                black_exp = new_black_exp.to_collector()
            else:
                white_exp = combine_experience(
                    [white_exp, new_white_exp.to_collector()], zero=True
                )
                black_exp = combine_experience(
                    [black_exp, new_black_exp.to_collector()], zero=True
                )
        else:
            white_exp = new_white_exp
            black_exp = new_black_exp

        white_loss_list.append(white_loss)
        black_loss_list.append(black_loss)

        # print(f"Attempting to join process {proc_id}")
        p = procs[proc_id]
        p.terminate()
        p.join()
        print(f"Finished sim process {proc_id}")

    logging.info(
        f"Average CP Loss: White {round(np.mean(white_loss_list))} | Black {round(np.mean(black_loss_list))}"
    )

    return white_exp, black_exp


def simulate_game(white_player, black_player, verbose=0, limit=True):
    game = chess.Board()
    agents: dict[bool, PrimeAgent] = {True: white_player, False: black_player}
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    while not game.is_game_over(claim_draw=True):
        color = game.turn
        if verbose:
            print(game)
            print("white to move" if color else "black to move")

        analysis = engine.analyse(
            game,
            limit=chess.engine.Limit(depth=10),
            info=chess.engine.INFO_ALL,
            game=object(),
            multipv=3,
        )

        cp_score = analysis[0]["score"].pov(color).score(mate_score=10000)
        next_move = agents[color].select_move(game)
        # next_move_cp_value = [
        #     x["score"].pov(color).score(mate_score=10000)
        #     for x in analysis
        #     if x["pv"][0] == next_move
        # ][0]
        # cp_loss = cp_score - next_move_cp_value

        # rename to reward if you remove the alt. reward below
        # reward = max(-1, 1 - (cp_loss / 100))

        # remove this later
        reward = np.clip((cp_score / 100), -1, 1)

        if agents[color].collector is not None:
            agents[color].collector.rewards.extend([reward])

        game.push(next_move)

        # analysis = engine.analyse(
        #     game,
        #     limit=chess.engine.Limit(depth=15),
        #     info=chess.engine.INFO_ALL,
        #     game=object(),
        #     multipv=1,
        # )
        # cp_score = analysis[0]["score"].pov(color).score(mate_score=10000)
        # reward = np.clip((cp_score / 100), -1, 1)
        # if agents[color].collector is not None:
        #     agents[color].collector.rewards.extend([reward])

        if verbose:
            print(next_move, end="\n\n")
        if limit and game.fullmove_number > 50:
            return None, game

    game_result = game.outcome(claim_draw=True)

    engine.quit()
    return game_result.winner, game


def evaluate_model(agent1, agent2, num_games, target, temperature, verbose=0):
    wins = 0
    losses = 0
    draws = 0
    played = 0
    color = True

    agent1.set_collector(None)
    agent2.set_collector(None)
    agent1.set_temperature(temperature)
    agent2.set_temperature(temperature)

    t = trange(num_games)
    for _ in t:
        if color:
            white_player, black_player = agent1, agent2
        else:
            black_player, white_player = agent1, agent2

        game_record, _ = simulate_game(
            white_player, black_player, verbose=verbose, limit=False
        )
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
            new_step = int(
                f.removeprefix("exp_")
                .removesuffix(".h5")
                .replace("_w", "")
                .replace("_b", "")
            )
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
    logging.basicConfig(
        filename=f"logs/{model_name}_training.log", encoding="utf-8", level=logging.INFO
    )
    logging.info("\n")
    logging.info(f"Start {model_name}_{model_version} Step {exp_step}")
    processes = 8

    num_games = 100
    eval_games = 10
    eval_search_rounds = 200
    eval_target = 0.60

    temperature = 1
    search_rounds = 100
    batch_size = 512

    if num_games < processes:
        num_games = processes
        print(
            "WARN: Number of games must be equal to or greater than number of processes"
        )

    model = models.load_model(
        f"dlchess/models/{model_name}_{model_version}", compile=(model_version > 0)
    )

    white_exp, black_exp = mp_simulate(
        num_games,
        processes,
        encoder,
        temperature,
        model_name,
        model_version,
        search_rounds,
    )

    white_exp.serialize(
        h5py.File(
            f"/data/code/botfjord/data/{model_name}_{model_version}/exp_{exp_step}_w.h5",
            mode="w",
        )
    )
    black_exp.serialize(
        h5py.File(
            f"/data/code/botfjord/data/{model_name}_{model_version}/exp_{exp_step}_b.h5",
            mode="w",
        )
    )

    if exp_step > 0:
        w_in_progress_model = models.load_model(
            f"dlchess/models/{model_name}_white_progress"
        )
        b_in_progress_model = models.load_model(
            f"dlchess/models/{model_name}_black_progress"
        )
        white_player = PrimeAgent(w_in_progress_model, encoder, eval_search_rounds)
        black_player = PrimeAgent(b_in_progress_model, encoder, eval_search_rounds)
    else:
        white_player = PrimeAgent(model, encoder, eval_search_rounds)
    # black_player = PrimeAgent(model, encoder, eval_search_rounds)

    # exp = load_experience(
    #     h5py.File(
    #         f"/data/code/botfjord/data/{model_name}_{model_version}/exp_{exp_step}.h5",
    #         mode="r",
    #     ),
    #     zero=True,
    # )
    # exp = check_lichess(exp)

    white_exp.shuffle()
    black_exp.shuffle()

    opt = SGD(learning_rate=0.02)
    white_player.opt = opt
    white_player.model.optimizer = None
    black_player.opt = opt
    black_player.model.optimizer = None

    white_player.train(white_exp, batch_size, loss_weights=[0.25, 1])
    white_player.serialize(f"dlchess/models/{model_name}_white_progress")
    black_player.train(black_exp, batch_size, loss_weights=[0.25, 1])
    black_player.serialize(f"dlchess/models/{model_name}_black_progress")

    if exp_step % 10 == 0 and exp_step > 0:
        white_player.serialize(f"dlchess/models/{model_name}_white_{exp_step // 10}")
        black_player.serialize(f"dlchess/models/{model_name}_black_{exp_step // 10}")
    #   if evaluate_model(
    #      white_player, black_player, eval_games, eval_target, temperature
    # ):
    #    white_player.serialize(f"dlchess/models/{model_name}_{model_version + 1}")
    #   result_queue.put(True)
    # else:
    #   result_queue.put(False)
    # else:
    #   result_queue.put(False)
    result_queue.put(False)


if __name__ == "__main__":
    model_name = "omega"
    model_version = 2
    target_version = 10

    logging.basicConfig(
        filename=f"logs/{model_name}_training.log", encoding="utf-8", level=logging.INFO
    )

    os.makedirs(f"/data/code/botfjord/data/{model_name}_{model_version}", exist_ok=True)

    encoder = OmegaEncoder()

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
