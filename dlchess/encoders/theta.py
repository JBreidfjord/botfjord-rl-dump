from __future__ import annotations

import numpy as np
from dlchess.encoders.base import Encoder
from dlchess.rl.policy_index import short_policy_index as policy_index

from chess import Board, Move


class ThetaEncoder(Encoder):
    def encode(self, game_state: Board):
        game_state = game_state.copy()
        pieces = list(range(1, 7)) * 2
        # Reverse sort (or not) to get current player's piece planes before opponent's
        colors = sorted([True, False] * 6, reverse=game_state.turn)
        arr: np.ndarray = np.zeros(shape=(8, 8, 56))
        # Records piece indices to array
        for i, (piece, color) in enumerate(zip(pieces, colors)):
            for j in game_state.pieces(piece, color):
                if not game_state.turn:
                    # Reverse indices to reorient board to black's POV
                    j = 63 - j
                arr[j >> 3, j & 7, i] = 1.0

        # Clean this up after, the board encoding could be a helper function
        try:
            temp_board = game_state.copy()
            temp_board.pop()
            for i, (piece, color) in enumerate(zip(pieces, colors)):
                i + 12
                for j in temp_board.pieces(piece, color):
                    if not game_state.turn:
                        j = 63 - j
                    arr[j >> 3, j & 7, i] = 1.0

            temp_board.pop()
            for i, (piece, color) in enumerate(zip(pieces, colors)):
                i + 24
                for j in temp_board.pieces(piece, color):
                    if not game_state.turn:
                        j = 63 - j
                    arr[j >> 3, j & 7, i] = 1.0

            temp_board.pop()
            for i, (piece, color) in enumerate(zip(pieces, colors)):
                i + 36
                for j in temp_board.pieces(piece, color):
                    if not game_state.turn:
                        j = 63 - j
                    arr[j >> 3, j & 7, i] = 1.0

        except IndexError:
            pass

        # Records two-fold repetition as warning
        if game_state.is_repetition(2):
            arr[:, :, -8] = 1.0

        # Records side to move and castling rights
        castling_rights = game_state.castling_xfen()
        if game_state.turn:
            arr[:, :, -7] = 1.0
            if "K" in castling_rights:
                arr[:, :, -5] = 1.0
            if "Q" in castling_rights:
                arr[:, :, -4] = 1.0
            if "k" in castling_rights:
                arr[:, :, -3] = 1.0
            if "q" in castling_rights:
                arr[:, :, -2] = 1.0
        else:
            if "k" in castling_rights:
                arr[:, :, -5] = 1.0
            if "q" in castling_rights:
                arr[:, :, -4] = 1.0
            if "K" in castling_rights:
                arr[:, :, -3] = 1.0
            if "Q" in castling_rights:
                arr[:, :, -2] = 1.0

        # Records move count and no progress count
        arr[:, :, -6] = game_state.fullmove_number
        arr[:, :, -1] = game_state.halfmove_clock

        return arr

    def encode_move(self, move: Move | str):
        if isinstance(move, Move):
            move = move.uci()
        return policy_index.index(move)

    def decode_move_index(self, move_idx: int):
        return Move.from_uci(policy_index[move_idx])

    def shape(self):
        return (8, 8, 56)

    def num_moves(self):
        return len(policy_index)

    def is_valid_move(self, move: Move | str):
        if isinstance(move, Move):
            move = move.uci()
        return move in policy_index
