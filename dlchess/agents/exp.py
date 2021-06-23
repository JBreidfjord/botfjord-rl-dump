from __future__ import annotations

import timeit

import numpy as np
import tensorflow as tf
from dlchess.agents.base import Agent
from dlchess.encoders.zero import ZeroEncoder
from dlchess.rl.experience import ExperienceBuffer, ExperienceCollector
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tqdm.auto import tqdm

import chess.engine
from chess import Board

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class TreeNode:
    def __init__(self, state: Board, value, priors, parent=None, last_move=None):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}

        for move in list(state.legal_moves):
            p = priors[move]
            self.branches[move] = Branch(p)

        self.children = {}

    def moves(self):
        """Returns a list of all possible moves from this node"""
        return self.branches.keys()

    def add_child(self, move, child_node):
        self.children[move] = child_node

    def has_child(self, move):
        return move in self.children

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, move):
        return self.branches[move].prior

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0

    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def check_visit_counts(self, num_rounds: int):
        visit_counts = sorted(
            self.branches.values(), key=lambda x: x.visit_count, reverse=True
        )
        remaining_rounds = num_rounds - self.total_visit_count
        if (
            visit_counts[0].visit_count
            >= visit_counts[1].visit_count + remaining_rounds
        ):
            return True
        return False


class ExpAgent(Agent):
    def __init__(
        self,
        encoder: ZeroEncoder,
        num_rounds: int = 1600,
        collector: ExperienceCollector = None,
    ):
        self.encoder = encoder
        self.collector = collector
        self.c = 5.0
        self.noise = 0.3
        self.num_rounds = num_rounds
        self.verbose = False

        self.engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    def set_collector(self, collector: ExperienceCollector):
        self.collector = collector

    def set_temperature(self, temperature: float):
        self.c = temperature

    def set_noise(self, noise: float = 0.3):
        self.noise = noise

    def set_verbosity(self, verbose: bool):
        self.verbose = verbose

    def select_branch(self, node: TreeNode, _list: bool = False):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + (self.c * p * np.sqrt(total_n) / (n + 1))

        if _list:
            return sorted(
                [(x.uci(), round(score_branch(x), 3)) for x in node.moves()],
                key=lambda x: x[1],
                reverse=True,
            )

        # Max will return the move with the highest score_branch value
        return max(node.moves(), key=score_branch)

    def select_move(self, game_state):
        # to remove timing:
        # this stuff; stuff before return; stuff in create_node around prediction
        self.prediction_time = 0.0
        self.run_time = 0.0
        start_time = timeit.default_timer()

        root = self.create_node(game_state)

        best = None
        last_change = None
        early_stop = None
        es_factor = 5
        t = tqdm(
            range(self.num_rounds),
            leave=False,
            ncols=80,
            disable=not self.verbose,
        )
        for i in t:
            node = root
            next_move = self.select_branch(node)

            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)

            new_state = node.state.copy()
            new_state.push(next_move)
            child_node = self.create_node(new_state, parent=node)

            move = next_move
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value = -1 * value

            # Check if only one legal move
            # No need to continue searching
            if len(root.moves()) == 1:
                break

            top_5 = sorted(root.moves(), key=root.visit_count, reverse=True)[:5]
            top_5 = [(x.uci(), root.visit_count(x)) for x in top_5]

            # scored = self.select_branch(node, _list=True)[:5]
            cb = top_5[0][0]
            if cb != best:
                best = cb
                last_change = i
                # t.set_description(best)

            try:
                t.set_description(
                    f"{top_5[0][0]} {top_5[0][1]} | {top_5[1][0]} {top_5[1][1]} | {top_5[2][0]} {top_5[2][1]}",
                    refresh=False,
                )
            except IndexError:
                t.set_description(
                    f"{top_5[0][0]} {top_5[0][1]} | {top_5[1][0]} {top_5[1][1]}",
                    refresh=False,
                )

            if (
                top_5[0][1] > (top_5[1][1] * es_factor)
                and i >= self.num_rounds // (es_factor * 2)
                and i >= 10
                # and cb == scored[0][0]
                and early_stop is None
            ):
                # early_stop = (top_5[0][0], i)
                break

            # print(i, next_move, cb, last_change, early_stop)
            # print(top_5)
            # print(scored)
            # print("")

            # Checks if any branch has more than half of the
            # max search rounds, making it the guaranteed choice
            if root.check_visit_counts(self.num_rounds):
                break

        if self.collector is not None:
            root_state_tensor = self.encoder.encode(game_state)
            visit_counts = np.array(
                [
                    root.visit_count(self.encoder.decode_move_index(idx))
                    for idx in range(1968)  # replace with num_moves()
                ]
            )
            assert np.max(visit_counts) > 0
            self.collector.record_decision(root_state_tensor, visit_counts)

        best_move = max(root.moves(), key=root.visit_count)

        if early_stop is not None and early_stop[0] != best_move.uci():
            print("EARLY STOP ERROR", early_stop, best_move.uci())
        elif early_stop is not None:
            print("Early stop successful", early_stop)
            print(f"Saved {self.num_rounds - early_stop[1]} searches")
            print(
                f"Decided: {last_change} | Confident: {early_stop[1]} | Max: {self.num_rounds}"
            )
        run_time = timeit.default_timer() - start_time - self.prediction_time
        # print(game_state.fullmove_number)
        if self.verbose:
            print(
                f"Prediction time: {self.prediction_time} | Calc time: {run_time} | {round((self.prediction_time / (self.prediction_time + run_time)) * 100, 2)}%"
            )
        return best_move

    def create_node(self, game_state, move=None, parent=None):
        pred_start_time = timeit.default_timer()

        color = game_state.turn

        def get_value(board):
            score = (
                self.engine.analyse(
                    board,
                    limit=chess.engine.Limit(depth=4),
                    info=chess.engine.INFO_SCORE,
                    game=object(),
                )["score"]
                .pov(color)
                .score(mate_score=100000)
                / 100
            )
            return score

        try:
            value = get_value(game_state)
        except:
            try:
                get_value(game_state)
            except:
                value = 0.0
        move_priors = {}
        for _move in list(game_state.legal_moves):
            board = game_state.copy(stack=False)
            board.push(_move)
            try:
                move_priors[_move] = get_value(board)
            except:
                move_priors[_move] = 0.0

        self.prediction_time += timeit.default_timer() - pred_start_time

        # Add Dirichlet noise
        if self.noise is not None:
            noise = np.random.dirichlet([self.noise] * len(list(move_priors.values())))
            for noise_val, (_move, prior) in zip(noise, move_priors.items()):
                move_priors[_move] = np.average([prior, noise_val], axis=0)

        new_node = TreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0
