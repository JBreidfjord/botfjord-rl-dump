from collections import namedtuple

import requests

import chess
import chess.syzygy


def probe_wdl(board: chess.Board):
    with chess.syzygy.open_tablebase("/data/code/botfjord/tablebase") as tb:
        wdl = tb.get_wdl(board)
        if wdl is None:
            id_str = get_id(board)
            get_files(id_str)
            wdl = tb.get_wdl(board)
        return wdl


def probe_move(board: chess.Board):
    best_move = None
    best_score = 0
    # do i need to minimax?
    # or can i just search current legal moves and take best


def get_id(board: chess.Board):
    fen = board.board_fen()
    Pieces = namedtuple("Pieces", ["queens", "rooks", "bishops", "knights", "pawns"])
    white = Pieces(
        fen.count("Q"), fen.count("R"), fen.count("B"), fen.count("N"), fen.count("P")
    )
    black = Pieces(
        fen.count("q"), fen.count("r"), fen.count("b"), fen.count("n"), fen.count("p")
    )
    if sum(white) + sum(black) + 2 > 7:
        return

    id_str: str = "K"
    id_str += "Q" * white.queens
    id_str += "R" * white.rooks
    id_str += "B" * white.bishops
    id_str += "N" * white.knights
    id_str += "P" * white.pawns

    id_str += "vK"

    id_str += "Q" * black.queens
    id_str += "R" * black.rooks
    id_str += "B" * black.bishops
    id_str += "N" * black.knights
    id_str += "P" * black.pawns

    if sum(white) < sum(black):
        swap = id_str.split("v")
        id_str = "v".join(swap[::-1])

    return id_str


def get_files(id_str: str):
    url = f"https://syzygy-tables.info/download/{id_str}.txt?source=lichess&dtz=root"
    url_list = requests.get(url).text.split()
    for file_url in url_list:
        filename = file_url.rsplit("/")[-1]
        r = requests.get(file_url)
        with open(f"/data/code/botfjord/tablebase/{filename}", mode="wb") as f:
            f.write(r.content)
