from __future__ import annotations

import math
import random

import numpy as np
from tensorflow.keras import models

import chess
import chess.engine
from dlchess.proc import get_position_array


class MCTSNode(object):
    def __init__(
        self, game_state: chess.Board, parent: MCTSNode = None, move: chess.Move = None
    ):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {chess.WHITE: 0, chess.BLACK: 0}
        self.num_rollouts = 0
        self.children: list[MCTSNode] = []
        self.unvisited_moves = list(game_state.legal_moves)

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.copy()
        new_game_state.push(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        # if winner is not None:
        #     winner = chess.WHITE if winner else chess.BLACK
        #     self.win_counts[winner] += 1
        # else:
        #     self.win_counts[chess.WHITE] += 0.5
        #     self.win_counts[chess.BLACK] += 0.5
        # self.num_rollouts += 1

        self.win_counts[chess.WHITE] += winner
        self.win_counts[chess.BLACK] += 1000 - winner
        self.num_rollouts += 1000
        # try:
        #     self.win_counts[chess.WHITE] += winner.wins
        #     self.win_counts[chess.BLACK] += winner.losses
        #     self.num_rollouts += winner.total()
        # except:  # add proper exception to remove bare except
        #     ...

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_state.is_game_over()

    def winning_pct(self, player: chess.Color):
        player = chess.WHITE if player else chess.BLACK
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MonteBot:
    def __init__(
        self,
        temperature: float = 1.5,
        max_search_rounds: int = 100,
        model: models.Model = None,
    ):
        self.temperature = temperature
        self.num_rounds = max_search_rounds
        self.model = model

    def select_move(self, game_state: chess.Board):
        root = MCTSNode(game_state)

        for _ in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node: MCTSNode = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_child()

            results = self.simulate_random_game(node.game_state)

            while node is not None:
                node.record_win(results)
                node = node.parent

        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_pct(game_state.turn)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        return best_move

    def select_child(self, node: MCTSNode):
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score = -1
        best_child = None
        for child in node.children:
            score = uct_score(
                total_rollouts,
                child.num_rollouts,
                child.winning_pct(node.game_state.turn),
                self.temperature,
            )
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def simulate_random_game(self, game_state: chess.Board) -> chess.Color:
        game = game_state.copy()
        # while not game.is_game_over():
        #     move = np.random.choice(list(game.legal_moves))
        #     game.push(move)
        # return game.outcome().winner
        if self.model is not None:
            # array/list nonsense makes it (1, 773)
            encoded_game = np.array([get_position_array(game)])
            pred = self.model(encoded_game)
            return np.round(pred[0, 0].numpy() * 1000)

        with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as engine:
            result = (
                engine.analyse(
                    game,
                    chess.engine.Limit(depth=10),
                    game=game,
                    info=chess.engine.INFO_SCORE,
                )
                .get("score")
                .white()
            )
            return result.wdl(model="lichess").expectation() * 1000


def uct_score(
    parent_rollouts: int, child_rollouts: int, win_pct: float, temperature: float
):
    exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_pct + (temperature * exploration)


def make_move(agent, move_string: str = None, board: chess.Board = None) -> str:
    """Main function to call, will handle a given game and return a move to execute"""
    if move_string is not None:
        board = chess.Board()
        for move in move_string.split():
            board.push_uci(move)

    return agent.select_move(board).uci()
