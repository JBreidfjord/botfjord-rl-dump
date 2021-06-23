import numpy as np
from dlchess.encoders.base import Encoder
from dlchess.rl.experience import ExperienceCollector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from chess import Board, Move


class QAgent:
    def __init__(self, model: Model, encoder: Encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0

    def set_temperature(self, temperature: float):
        self.temperature = temperature

    def set_collector(self, collector: ExperienceCollector):
        self.collector = collector

    def select_move(self, game_state: Board):
        board_tensor = self.encoder.encode(game_state)

        if np.random.random() < self.temperature:
            move = np.random.choice(list(game_state.legal_moves))
            if self.collector is not None:
                encoded_move = self.encoder.encode_move(move)
                self.collector.record_decision(state=board_tensor, action=encoded_move)
            return move

        moves = []
        board_tensors = []
        for move in game_state.legal_moves:
            moves.append(self.encoder.encode_move(move))
            board_tensors.append(board_tensor)

        num_moves = len(moves)
        board_tensors = np.array(board_tensors)
        move_src_vectors = np.zeros((num_moves, 64))
        move_dst_vectors = np.zeros((num_moves, 64))
        for i, move in enumerate(moves):
            move_src_vectors[i][move[0]] = 1.0
            move_dst_vectors[i][move[1]] = 1.0

        values = self.model.predict([board_tensors, move_src_vectors, move_dst_vectors])
        # Reshapes from (1, n) to (n,)
        values = values.reshape(len(moves))

        ranked_moves = np.argsort(values)
        ranked_moves = np.flip(ranked_moves)

        for move_idx in ranked_moves:
            decoded_move: Move = self.encoder.decode_move_indices(moves[move_idx])
            if game_state.is_legal(decoded_move):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=moves[move_idx],
                        estimated_value=values[move_idx],
                    )
                return decoded_move

        # Catch if None falls through
        move = np.random.choice(list(game_state.legal_moves))
        if self.collector is not None:
            encoded_move = self.encoder.encode_move(move)
            self.collector.record_decision(state=board_tensor, action=encoded_move)
        return move

    def train(self, experience, lr=0.1, batch_size=128, epochs=1):
        opt = SGD(lr=lr)
        self.model.compile(loss="mse", optimizer=opt)

        n = experience.states.shape[0]
        y = np.zeros((n,))
        actions_src = np.zeros((n, 64))
        actions_dst = np.zeros((n, 64))
        for i in range(n):
            action_src, action_dst = experience.actions[i]
            reward = experience.rewards[i]
            actions_src[i][action_src] = 1.0
            actions_dst[i][action_dst] = 1.0
            y[i] = reward

        self.model.fit(
            [experience.states, actions_src, actions_dst],
            y,
            batch_size=batch_size,
            epochs=epochs,
        )

    def serialize(self, filepath):
        self.model.save(filepath)
