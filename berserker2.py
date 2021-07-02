import json
import threading
import time

import berserk
import h5py
import requests
import tensorflow as tf
from dlchess.agents.base import Agent
from dlchess.agents.prime import PrimeAgent
from dlchess.bot.config import config
from dlchess.encoders.base import Encoder
from dlchess.encoders.theta import ThetaEncoder
from dlchess.rl.experience import ZeroCollector
from tensorflow.keras import models

import chess

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

session = berserk.TokenSession(config["LICHESS_TOKEN"])
client = berserk.Client(session)


class Game(threading.Thread):
    def __init__(
        self,
        client: berserk.Client,
        game_id: str,
        **kwargs,
    ):
        self._is_running = True
        super().__init__(**kwargs)

        encoder = ThetaEncoder()
        model = models.load_model("dlchess/models/theta_large_progress")
        self.collector = ZeroCollector()
        self.collector.begin_episode()
        self.agent = PrimeAgent(
            model,
            encoder,
            num_rounds=800,
            collector=self.collector,
            prevent_repetition=True,
        )
        self.agent.set_verbosity(True)
        # self.agent.set_temperature(3.5)

        self._is_searching = False

        self.game_id = game_id
        self.client = client
        self.stream = self.client.bots.stream_game_state(game_id)

        self.board = None
        self.get_game_state()

        self.name = client.account.get()["id"]
        self.chat_active = True
        self.white = self.name if self.color == "white" else self.opponent["username"]
        self.black = self.name if self.color == "black" else self.opponent["username"]
        print(f"Game {self.game_id} | Initialized | {self.white} v {self.black}")

    def run(self):
        while self._is_running:
            for event in self.stream:
                if event["type"] == "gameFull":
                    self.initial_fen = (
                        event["initialFen"]
                        if event["initialFen"] != "startpos"
                        else chess.STARTING_FEN
                    )
                    self.board = chess.Board(self.initial_fen)
                    event = event["state"]

                if event["type"] == "gameState":
                    self._is_searching = True
                    self.board = chess.Board(self.initial_fen)
                    for uci_move in event["moves"].split():
                        self.board.push_uci(uci_move)
                    self._is_searching = False

                time.sleep(0.2)
                self.get_game_state()
                self.check_turn()

        if not self.collector.is_empty():
            game_result: dict = self.client.games.export(self.game_id)
            # Update to match statement in 3.10
            game_winner = game_result.get("winner")
            if game_winner is None:
                self.collector.complete_episode(reward=0)
            elif game_winner == self.color:
                self.collector.complete_episode(reward=1)
            elif game_winner == self.op_color:
                self.collector.complete_episode(reward=-1)
            else:
                raise ValueError("Game winner was not valid")

            self.collector.multiply(10)
            experience = self.collector.to_buffer()
            experience.serialize(
                h5py.File(
                    f"/data/code/botfjord/data/lichess/{self.name}_{self.game_id}.h5",
                    mode="w",
                )
            )

    def stop(self):
        print(f"Game {self.game_id} | Exited")
        self._is_running = False

    def get_game_state(self):
        for game in self.client.games.get_ongoing(count=100):
            if game["gameId"] == self.game_id:
                self.game = game
                self.fen = self.game["fen"]
                self.color = self.game["color"]
                self.op_color = "white" if self.color == "black" else "black"
                self.opponent = self.game["opponent"]
                self.is_my_turn = self.game["isMyTurn"]
                self.last_move = self.game["lastMove"]
                return
        self.stop()  # Exit if none

    def check_turn(self):
        self.get_game_state()
        if self.is_my_turn and not self._is_searching and not self.board.is_game_over():
            self._is_searching = True
            print(
                f"Game {self.game_id} | {self.opponent['username']} ({self.op_color}) | {self.last_move}\n"
            )
            next_move = get_move(
                self.agent, self.board, fen_string=self.fen, turn=self.color == "white"
            )
            try:
                client.bots.make_move(game_id=self.game_id, move=next_move)
            except:
                self._is_searching = False
                return
            time.sleep(0.5)
            self.get_game_state()
            print(
                f"Game {self.game_id} | {self.name} ({self.color}) | {self.last_move}\n"
            )
            self._is_searching = False

    def handle_chat_line(self, chat_line):
        if self.chat_active:
            client.bots.post_message(
                game_id=self.game_id, text="Sorry, I'm not set up for chat yet!"
            )
            self.chat_active = False
        self.get_game_state()


class Search(threading.Thread):
    def __init__(self, agent, fen_string, turn, **kwargs):
        self._is_running = True
        super().__init__(**kwargs)

        self.agent = agent
        self.board = chess.Board(fen_string)
        self.board.turn = turn

    def run(self):
        while self._is_running:
            for opponent_move in list(self.board.legal_moves):
                ...


accept_players = [
    "jbreidfjord",
    "therudeduck",
    "botfjordslayer69",
    "botfjord",
    "anonymous",
    "botfjord-dev",
]

games: list[Game] = []


def get_syzygy(board: chess.Board) -> chess.Move:
    time.sleep(1)
    base_url = "http://tablebase.lichess.ovh/standard/mainline?fen="
    fen = board.fen().replace(" ", "_")
    r = requests.get(base_url + fen)
    if r.status_code != 200:
        if r.status_code == 429:
            time.sleep(1)
            return
        return
    r = json.loads(r.content)
    if not r["mainline"]:
        return
    return r["mainline"][0]["uci"]


def get_move(agent, game_state, fen_string: str, turn: bool) -> str:
    if game_state is None:
        board = chess.Board(fen_string)
        board.turn = turn
    else:
        board = game_state
    if len(board.piece_map()) <= 7:
        tb_move = get_syzygy(board)
        if tb_move is not None:
            return tb_move
    return agent.select_move(board).uci()


def auto_check():
    ongoing = client.games.get_ongoing(count=100)

    for game in games:
        if game.game_id not in [game_info["gameId"] for game_info in ongoing]:
            game.stop()
            time.sleep(0.1)
            games.remove(game)

    for game in games:
        if not game._is_searching:
            game.check_turn()

    t = threading.Timer(5, auto_check)
    t.start()


def should_accept(event) -> bool:
    # event['challenge']['variant']['name'] == 'Standard' and
    if event["challenge"]["challenger"]["id"].lower() in accept_players:
        return True
    else:
        return False


auto_check()
is_polite = True
for event in client.bots.stream_incoming_events():
    # print(event)
    if event["type"] == "challenge":
        if should_accept(event):
            try:
                client.bots.accept_challenge(event["challenge"]["id"])
            except:
                continue
        elif is_polite:
            try:
                client.bots.decline_challenge(event["challenge"]["id"])
            except:
                continue
    elif event["type"] == "gameStart":
        if event["game"]["id"] not in [game.game_id for game in games]:
            game = Game(client=client, game_id=event["game"]["id"])
            print(f"Game {game.game_id} | Start")
            games.append(game)
            game.start()
            time.sleep(0.1)

    ongoing = client.games.get_ongoing(count=100)

    for game_info in ongoing:
        if game_info["gameId"] not in [game.game_id for game in games]:
            game = Game(client=client, game_id=game_info["gameId"])
            print(f"Game {game.game_id} | Force Start")
            games.append(game)
            game.start()
            time.sleep(0.1)
