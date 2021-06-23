from dlchess.agents.prime import PrimeAgent
from dlchess.encoders.theta import ThetaEncoder
from rl_train import evaluate_model
from tensorflow.keras.models import load_model

num_games = 5

# input paths and agents
new_model = load_model("dlchess/models/theta_small_1")
old_model = load_model("dlchess/models/theta_small_1")
encoder = ThetaEncoder()

agent1 = PrimeAgent(new_model, encoder, 500)
agent2 = PrimeAgent(old_model, encoder, 500)
agent1.set_verbosity(True)

agent1.prevent_repetition = True
agent2.prevent_repetition = True

wins = 0
losses = 0
draws = 0
color = True

evaluate_model(agent1, agent2, num_games, 0.6, verbose=1)
