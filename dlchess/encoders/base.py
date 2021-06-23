from numpy import ndarray

from chess import Board, Move


class Encoder:
    def name(self):
        raise NotImplementedError()

    def encode(self, game_state: Board) -> ndarray:
        raise NotImplementedError()

    def encode_move(self, uci_move: str = None, move: Move = None) -> tuple[int, int]:
        raise NotImplementedError()

    def decode_move_indices(self, src_idx: int, dst_idx: int) -> Move:
        raise NotImplementedError()

    def shape(self) -> tuple[int]:
        raise NotImplementedError()
