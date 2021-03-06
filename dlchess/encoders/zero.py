from __future__ import annotations

import numpy as np
from chess import Board, Move
from dlchess.encoders.base import Encoder
from dlchess.rl.policy_index import short_policy_index as policy_index
from tensorflow.keras.models import Model


class ZeroEncoder(Encoder):
    def __init__(self, encoder_model: Model = None):
        self.encoder_model = encoder_model
        if self.encoder_model is not None:
            for layer in self.encoder_model.layers:
                layer.trainable = False

        self.game_hash: dict[str, np.ndarray] = {}

    def encode(self, game_state: Board) -> np.ndarray:
        encoded_arr: np.ndarray | None = self.game_hash.get(game_state.epd())

        if encoded_arr is None:
            pieces = list(range(1, 7)) * 2
            # If you just reverse this sort based on game turn,
            # would that reorient the board? No but it makes the current player planes come first
            # would just need to make j = 63 - j when turn is black
            colors = sorted([True, False] * 6)
            arr = np.zeros(shape=(1, 8, 8, 17))
            # Records piece indices to array
            for i, (piece, color) in enumerate(zip(pieces, colors)):
                for j in game_state.pieces(piece, color):
                    arr[:, j >> 3, j & 7, i] = 1.0
            # Add repetitions, total move count, and no progress counts if possible
            # Repetitions are 2 planes
            # Records castling rights to array
            castling_rights = game_state.castling_xfen()
            if "K" in castling_rights:
                arr[:, :, :, -5] = 1.0
            if "Q" in castling_rights:
                arr[:, :, :, -4] = 1.0
            if "k" in castling_rights:
                arr[:, :, :, -3] = 1.0
            if "q" in castling_rights:
                arr[:, :, :, -2] = 1.0

            # Records side to move to array
            if game_state.turn:
                arr[:, :, :, -1] = 1.0

            if self.encoder_model is not None:
                encoded_arr = self.encoder_model(arr, training=False)
            else:
                encoded_arr = arr

            self.game_hash[game_state.epd()] = encoded_arr

        return encoded_arr[-1]

    def encode_move(self, move: Move | str):
        if isinstance(move, Move):
            move = move.uci()
        return policy_index.index(move)

    def decode_move_index(self, move_idx: int):
        return Move.from_uci(policy_index[move_idx])

    def shape(self):
        if self.encoder_model is not None:
            return self.encoder_model.output_shape[-1]
        return (8, 8, 17)

    def num_moves(self):
        return len(policy_index)

    def is_valid_move(self, move: Move | str):
        if isinstance(move, Move):
            move = move.uci()
        return move in policy_index
