import numpy as np
import tensorflow as tf
from dlchess.encoders.base import Encoder
from dlchess.rl.experience import ExperienceBuffer
from dlchess.rl.policy_index import policy_index
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from chess import Board, Move

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_virtual_device_configuration(
#     physical_devices[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)],
# )
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)


class ActorCriticAgent:
    def __init__(self, model: Model, encoder: Encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 0.0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state: Board):
        num_moves = len(policy_index)

        board_tensor = self.encoder.encode(game_state)
        X = np.array([board_tensor])

        # with tf.device("/device:gpu:1"):
        actions, values = self.model.predict(X)
        move_probs = actions[0]
        estimated_value = values[0][0]

        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs /= np.sum(move_probs)

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs
        )
        for move_idx in ranked_moves:
            move = policy_index[move_idx]
            if game_state.is_legal(Move.from_uci(move)):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=move_idx,
                        estimated_value=estimated_value,
                    )
                return Move.from_uci(move)

    def train(
        self,
        experience: ExperienceBuffer,
        lr: float = 0.01,
        batch_size: int = 128,
        epochs: int = 1,
    ):
        opt = SGD(lr=lr)
        self.model.compile(
            loss=["categorical_crossentropy", "mse"],
            loss_weights=[1, 0.5],
            optimizer=opt,
        )

        n = experience.states.shape[0]
        policy_targets = np.zeros(shape=(n, len(policy_index)))
        value_targets = np.zeros(shape=(n,))
        for i in range(n):
            action = experience.actions[i]
            policy_targets[i][action] = experience.advantages[i]
            value_targets[i] = experience.rewards[i]

        self.model.fit(
            experience.states,
            [policy_targets, value_targets],
            batch_size=batch_size,
            epochs=epochs,
        )

    def serialize(self, filepath):
        self.model.save(filepath)
