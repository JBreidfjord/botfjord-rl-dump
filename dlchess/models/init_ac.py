from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, concatenate
from tensorflow.keras.models import Model

board_input = Input(shape=(8, 8, 17), name="board_input")

x = Conv2D(64, (3, 3), padding="same", activation="relu")(board_input)
x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)

x = Flatten()(x)
processed_board = Dense(512)(x)

policy_hidden_layer = Dense(512, activation="relu")(processed_board)
policy_output = Dense(1968, activation="softmax", name="policy")(policy_hidden_layer)

value_hidden_layer = Dense(512, activation="relu")(processed_board)
value_output = Dense(1, activation="tanh", name="value")(value_hidden_layer)

model = Model(board_input, [policy_output, value_output])

model.save("ac.h5", include_optimizer=False)
