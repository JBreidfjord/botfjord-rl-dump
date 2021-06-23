import argparse
import os
from secrets import token_hex

import h5py
import numpy as np
from dlchess.encoders.omega import OmegaEncoder
from dlchess.encoders.theta import ThetaEncoder
from dlchess.rl.experience import ZeroCollector
from tqdm.auto import tqdm

import chess
import chess.engine
import chess.pgn


class SLGen:
    def __init__(self):
        self.data_dir = None
        self.file_num = None

        self._omega_encoder = OmegaEncoder()
        self._theta_encoder = ThetaEncoder()
        self._engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

        self._games = self._game_generator()

    def _process_game(self, board: chess.Board):
        color = board.turn
        omega_input_array = self._omega_encoder.encode(board)
        theta_input_array = self._theta_encoder.encode(board)

        if color:
            assert (
                np.all(omega_input_array[:, :, -7]) == 1.0
                and np.all(theta_input_array[:, :, -7]) == 1.0
            )

        else:
            assert (
                np.all(omega_input_array[:, :, -7]) == 0.0
                and np.all(theta_input_array[:, :, -7]) == 0.0
            )

        analysis = self._engine.analyse(
            board,
            limit=chess.engine.Limit(depth=10),
            info=chess.engine.INFO_ALL,
            game=object(),
            multipv=board.legal_moves.count(),
        )

        if analysis is None:
            self._engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
            return

        policy_target: np.ndarray = np.zeros(
            (self._theta_encoder.num_moves()),
        )
        # Collect scores
        for res in analysis:
            value = res["score"].pov(color).score(mate_score=10000)
            # Normalize value to [0, 20000] to remove negatives before sum
            mv = res["pv"][0]
            if not self._theta_encoder.is_valid_move(mv):
                continue
            mv_idx = self._theta_encoder.encode_move(res["pv"][0])
            policy_target[mv_idx] = value

        policy_target[policy_target != 0.0] += np.max(np.abs(policy_target)) + 1

        # Normalize scores to distribution over [0,1] with a sum of 1
        policy_target /= np.sum(policy_target, initial=1e-10)
        try:
            assert np.allclose(np.sum(policy_target), [1])
        except AssertionError:
            print(f"AssertionError when {np.sum(policy_target)}")
            return

        # Gets expected value then normalizes to [-1, 1]
        # value_target = (
        #     analysis[0]["score"].pov(color).wdl(ply=board.ply()).expectation() - 0.5
        # ) / 0.5
        # assert -1 <= value_target <= 1
        value_target = analysis[0]["score"].pov(color).score(mate_score=10000)
        assert -10000 <= value_target <= 10000

        return omega_input_array, theta_input_array, policy_target, value_target

    def _game_generator(self):
        assert self.file_num is not None
        assert self.data_dir is not None
        self.file_num = int(self.file_num)

        pgn_files = [f for f in os.listdir(self.data_dir) if f.endswith(".pgn")]
        pgn_file = pgn_files[self.file_num]

        with open(os.path.join(self.data_dir, pgn_file), mode="r") as pgn:
            _active = True
            while _active:
                pgn_game = chess.pgn.read_game(pgn)
                if pgn_game is not None:
                    yield pgn_game
                else:
                    _active = False

    def _position_generator(self):
        game = next(self._games, None)
        if game is None:
            return
        positions = []
        board = game.board()
        positions.append(board)
        for move in game.mainline_moves():
            board = board.copy()
            board.push(move)
            if not board.is_game_over():
                positions.append(board)

        return positions

    def _position_batcher(self, batch_size: int = 5000):
        boards = []
        while len(boards) < batch_size:
            positions = self._position_generator()
            if positions is None:
                return
            boards.extend(positions)
        return boards

    def generate_experience(self, batch_size: int = 5000):
        positions = self._position_batcher(batch_size)
        if positions is None:
            return
        game_data = [self._process_game(x) for x in tqdm(positions)]

        omega_states, theta_states, policy_targets, reward_targets = [], [], [], []
        for game in game_data:
            if game is None:
                continue

            omega_states.append(game[0])
            theta_states.append(game[1])
            policy_targets.append(game[2])
            reward_targets.append(game[3])

        # Normalize rewards
        reward_targets = np.array(reward_targets, dtype="float32")
        reward_targets /= np.max(np.abs(reward_targets))
        reward_targets = reward_targets.tolist()

        omega_collector = ZeroCollector()
        theta_collector = ZeroCollector()
        omega_collector.set_data(
            omega_states, policy_targets.copy(), reward_targets.copy()
        )
        theta_collector.set_data(
            theta_states, policy_targets.copy(), reward_targets.copy()
        )
        omega_collector.shuffle()
        theta_collector.shuffle()

        return omega_collector.to_buffer(), theta_collector.to_buffer()


if __name__ == "__main__":
    gen = SLGen()

    parser = argparse.ArgumentParser(description="Generate chess training data")
    parser.add_argument("--file_num")
    parser.add_argument("--data_dir")
    parser.add_argument("--output")
    parser.parse_args(namespace=gen)
    args = parser.parse_args()
    print(args.output)

    _active = True
    while _active:
        omega_data, theta_data = gen.generate_experience()
        if omega_data is None or theta_data is None:
            _active = False
        else:
            filename = token_hex(8)
            omega_data.serialize(
                h5py.File(f"{args.output}/omega/{filename}.h5", mode="w")
            )
            theta_data.serialize(
                h5py.File(f"{args.output}/theta/{filename}.h5", mode="w")
            )
