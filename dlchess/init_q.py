from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model

board_input = Input(shape=(773,))
move_src_input = Input(shape=(64,))
move_dst_input = Input(shape=(64,))

board_and_action = concatenate([board_input, move_src_input, move_dst_input])
x = Dense(512, activation="relu")(board_and_action)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
value_output = Dense(1, activation="tanh")(x)

model = Model([board_input, move_src_input, move_dst_input], value_output)

model.save("q_0.h5", include_optimizer=False)
