import numpy as np

import chess


def get_position_array(board: chess.Board) -> np.ndarray:
    pieces = list(range(1, 7)) * 2
    colors = sorted([True, False] * 6)
    arr = np.zeros(shape=(773,))
    # Records piece indices to array
    for i, (piece, color) in enumerate(zip(pieces, colors)):
        for j in board.pieces(piece, color):
            arr[(i * 64) + j] = 1.0

    # Records castling rights to array
    castling_rights = board.castling_xfen()
    if "K" in castling_rights:
        arr[-5] = 1.0
    if "Q" in castling_rights:
        arr[-4] = 1.0
    if "k" in castling_rights:
        arr[-3] = 1.0
    if "q" in castling_rights:
        arr[-2] = 1.0

    # Records side to move to array
    if board.turn:
        arr[-1] = 1.0

    return arr
