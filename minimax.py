import chess


def minimax(
    position: chess.Board,
    depth: int,
    maximizing_player: bool,
    return_move: bool = False,
):
    if depth == 0 or position.is_game_over():
        return position.is_checkmate()

    if maximizing_player:
        mate_moves = []
        for move in position.legal_moves:
            board = position.copy(stack=False)
            board.push(move)
            mate = minimax(board, depth - 1, False)
            if mate:
                mate_moves.append(move)

        if mate_moves:
            return mate_moves if return_move else True
        else:
            return None if return_move else False

    else:
        mate_moves = []
        for move in position.legal_moves:
            board = position.copy(stack=False)
            board.push(move)
            mate = minimax(board, depth - 1, True)
            if mate:
                mate_moves.append(move)

        if mate_moves:
            return mate_moves if return_move else True
        else:
            return None if return_move else False
