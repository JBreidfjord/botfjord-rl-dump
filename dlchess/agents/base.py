from dlchess.encoders.base import Encoder
from dlchess.rl.experience import ExperienceBuffer, ExperienceCollector
from tensorflow.keras.models import Model


class Agent:
    def __init__(self, model: Model, encoder: Encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None

    def set_collector(self, collector: ExperienceCollector):
        self.collector = collector

    def select_move(self, game_state):
        raise NotImplementedError

    def train(
        self,
        experience: ExperienceBuffer,
        batch_size: int,
        epochs: int,
        loss_weights: list[int],
        verbose: bool,
    ):
        raise NotImplementedError

    def serialize(self, filepath):
        self.model.save(filepath)
