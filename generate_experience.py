import h5py
from dlchess.encoders.twelveplane import TwelvePlaneEncoder
from dlchess.rl.experience import ExperienceCollector, combine_experience
from dlchess.rl.q import QAgent
from tensorflow.keras import models

import chess


def simulate_game(white_player, black_player):
    game = chess.Board()
    agents = {True: white_player, False: black_player}

    while not game.is_game_over():
        next_move = agents[game.turn].select_move(game)
        game.push(next_move)
    game_result = game.outcome()

    if __name__ == "__main__":
        print(game)
        print([x.uci() for x in game.move_stack])
        print(game_result)
        print("")

    return game_result.winner


if __name__ == "__main__":
    num_games = 10
    temperature = 0.5
    model_name = "q"
    encoder_name = "full_encoder"
    experience_filename = "experience.h5"

    model = models.load_model("dlchess/models/" + model_name + ".h5", compile=False)
    encoder_model = models.load_model(
        "dlchess/models/" + encoder_name + ".h5", compile=False
    )
    encoder = TwelvePlaneEncoder(encoder_model)

    white_player = QAgent(model, encoder)
    black_player = QAgent(model, encoder)

    white_collector = ExperienceCollector()
    black_collector = ExperienceCollector()

    white_player.set_collector(white_collector)
    black_player.set_collector(black_collector)

    white_player.set_temperature(temperature)
    black_player.set_temperature(temperature)

    for i in range(num_games):
        white_collector.begin_episode()
        black_collector.begin_episode()

        try:
            game_record = simulate_game(white_player, black_player)
        except AttributeError:
            white_collector.reset_episode()
            black_collector.reset_episode()
            continue
        if game_record is None:
            white_collector.complete_episode(reward=0)
            black_collector.complete_episode(reward=0)
        elif game_record:
            white_collector.complete_episode(reward=1)
            black_collector.complete_episode(reward=-1)
        else:
            black_collector.complete_episode(reward=1)
            white_collector.complete_episode(reward=-1)

    experience = combine_experience([white_collector, black_collector])
    with h5py.File(experience_filename, "w") as experience_out:
        experience.serialize(experience_out)
