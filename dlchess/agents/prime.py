from __future__ import annotations

import timeit

import numpy as np
import tensorflow as tf
from dlchess.agents.base import Agent
from dlchess.encoders.prime import PrimeEncoder
from dlchess.rl.experience import ExperienceBuffer, ExperienceCollector
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tqdm.auto import tqdm

from chess import Board

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class PrimeTreeNode:
    def __init__(self, state: Board, value, priors, parent=None, last_move=None):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}

        for move in list(state.legal_moves):
            if move.uci().endswith("r") or move.uci().endswith("b"):
                continue
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


class PrimeAgent(Agent):
    def __init__(
        self,
        model: Model,
        encoder: PrimeEncoder,
        num_rounds: int = 1600,
        collector: ExperienceCollector = None,
        prevent_repetition: bool = False,
    ):
        self.model = model
        self.encoder = encoder
        self.collector = collector
        self.c = 5.0
        self.noise = 0.3
        self.num_rounds = num_rounds
        self.verbose = False

        self.prevent_repetition = prevent_repetition
        self.move_history = []

        # replaces predict(), ~5x speed increase on 3-256 model
        self.get_output = K.function(self.model.inputs, self.model.outputs)

        # Need to store optimizer state
        self.opt = (
            self.model.optimizer if self.model.optimizer else RMSprop(learning_rate=0.2)
        )

    def set_collector(self, collector: ExperienceCollector):
        self.collector = collector

    def set_temperature(self, temperature: float):
        self.c = temperature

    def set_noise(self, noise: float = 0.3):
        self.noise = noise

    def set_verbosity(self, verbose: bool):
        self.verbose = verbose

    def select_branch(self, node: PrimeTreeNode, _list: bool = False):
        total_n = node.total_visit_count

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + (self.c * p * (np.sqrt(total_n) / (n + 1)))

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
        game_state = game_state.copy()
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
            top_5 = [(game_state.san(x), root.visit_count(x)) for x in top_5]

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
                    for idx in range(self.encoder.num_moves())
                ]
            )
            assert np.max(visit_counts) > 0
            self.collector.record_decision(root_state_tensor, visit_counts)

        best_move = max(root.moves(), key=root.visit_count)
        if self.prevent_repetition:
            if len(self.move_history) >= 2 and len(root.moves()) >= 2:
                if (
                    best_move.uci() == self.move_history[-2]
                    and best_move.uci()[2:] + best_move.uci()[:2]
                    == self.move_history[-1]
                ):
                    old_best = best_move
                    # prevent repetition, grab 2nd best move
                    best_move = sorted(root.moves(), key=root.visit_count)[-2]
                    if self.verbose:
                        print(
                            f"Prevented repeating {old_best.uci()} with {best_move.uci()}"
                        )

        self.move_history.append(best_move.uci())

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
            if len(root.moves()) > 1:
                try:
                    print(
                        f"{top_5[0][0]} {top_5[0][1]} | {top_5[1][0]} {top_5[1][1]} | {top_5[2][0]} {top_5[2][1]}"
                    )
                except IndexError:
                    print(f"{top_5[0][0]} {top_5[0][1]} | {top_5[1][0]} {top_5[1][1]}")
            print(
                f"Prediction time: {self.prediction_time} | Calc time: {run_time} | {round((self.prediction_time / (self.prediction_time + run_time)) * 100, 2)}%"
            )
        return best_move

    def create_node(self, game_state: Board, move=None, parent=None):
        state_tensor = self.encoder.encode(game_state)
        model_input = np.array([state_tensor])

        pred_start_time = timeit.default_timer()
        # priors, values = self.model.predict(model_input)
        priors, values = self.get_output(model_input)
        self.prediction_time += timeit.default_timer() - pred_start_time

        priors = priors[0]
        value = values[0][0]

        # Mask non-legal moves then renormalize the distribution
        legal_indices = [
            self.encoder.encode_move(mv)
            for mv in game_state.legal_moves
            if self.encoder.is_valid_move(mv)
        ]
        mask = np.ones(priors.shape, dtype=bool)
        mask[legal_indices] = False
        priors[mask] = 0.0
        priors /= np.sum(priors, initial=1e-10)

        # Add Dirichlet noise
        if self.noise is not None:
            noise = np.random.dirichlet([self.noise] * len(legal_indices))
            noise_map = np.zeros(priors.shape)
            for noise_value, i in zip(noise, legal_indices):
                noise_map[i] = noise_value
            priors = np.average([priors, noise_map], axis=0, weights=[0.5, 0.5])

        move_priors = {
            self.encoder.decode_move_index(idx): p for idx, p in enumerate(priors)
        }
        new_node = PrimeTreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node

    def train(
        self,
        experience: ExperienceBuffer,
        batch_size=1024,
        epochs=1,
        loss_weights=[1, 1],
        verbose=1,
        ignore_draws: bool = False,
        validation_split=0.0,
    ):
        num_samples = experience.states.shape[0]
        model_inputs = experience.states
        # try:
        #     assert model_inputs.shape[-1] == self.encoder.shape()[-1]
        # except AssertionError:
        #     print(
        #         f"Invalid model input shape: Expected {self.encoder.shape()} received {model_inputs.shape[1:]}"
        #     )
        #     return

        # Normalize visit counts so they sum to 1
        visit_sums = np.sum(experience.visit_counts, axis=1).reshape((num_samples, 1))
        policy_targets = experience.visit_counts / visit_sums

        value_targets = experience.rewards

        if ignore_draws:
            indices = np.nonzero(value_targets)[0]
            model_inputs = model_inputs[indices]
            policy_targets = policy_targets[indices]
            value_targets = value_targets[indices]

            if value_targets.size <= (batch_size * 3):
                print("No training data")
                return

        if self.model.optimizer is None:
            # Avoid recompiling
            print("Model compiled")
            self.model.compile(
                # Adam(learning_rate=learning_rate, amsgrad=True, epsilon=1e-3),
                self.opt,
                loss=["categorical_crossentropy", "mse"],
                loss_weights=loss_weights,
            )

        return self.model.fit(
            model_inputs,
            [policy_targets, value_targets],
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
        )

    def serialize(self, filepath):
        self.model.save(filepath)


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0
