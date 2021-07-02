from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

num_blocks = 5
filter_size = 128
input_size = (8, 8, 20)


def build_res_block(x, idx):
    res_input = x
    res_name = "res" + str(idx)

    x = Conv2D(
        filters=filter_size,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(1e-4),
        name=res_name + "_conv_1",
    )(x)
    x = BatchNormalization(axis=-1, name=res_name + "_batch_norm_1")(x)
    x = Activation("relu", name=res_name + "_activation_1")(x)

    x = Conv2D(
        filters=filter_size,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(1e-4),
        name=res_name + "_conv_2",
    )(x)
    x = BatchNormalization(axis=-1, name=res_name + "_batch_norm_2")(x)
    x = Add(name=res_name + "_add")([res_input, x])
    x = Activation("relu", name=res_name + "_activation_2")(x)

    return x


board_input = x = Input(shape=input_size, name="board_input")

x = Conv2D(
    filters=filter_size,
    kernel_size=5,
    padding="same",
    use_bias=False,
    kernel_regularizer=l2(1e-4),
    name="input_conv",
)(x)
x = BatchNormalization(axis=-1, name="input_batch_norm")(x)
x = Activation("relu", name="input_activation")(x)

for i in range(num_blocks):
    x = build_res_block(x, i + 1)

res_output = x

# Policy Head
x = Conv2D(
    filters=2,
    kernel_size=1,
    use_bias=False,
    kernel_regularizer=l2(1e-4),
    name="policy_conv",
)(res_output)
x = BatchNormalization(axis=-1, name="policy_batch_norm")(x)
x = Activation("relu", name="policy_activation")(x)
x = Flatten(name="policy_flatten")(x)
policy_output = Dense(
    1880,
    kernel_regularizer=l2(1e-4),
    activation="softmax",
    name="policy_output",
)(x)

# Value Head
x = Conv2D(
    filters=4,
    kernel_size=1,
    use_bias=False,
    kernel_regularizer=l2(1e-4),
    name="value_conv",
)(res_output)
x = BatchNormalization(axis=-1, name="value_batch_norm")(x)
x = Activation("relu", name="value_activation")(x)
x = Flatten(name="value_flatten")(x)
x = Dense(256, kernel_regularizer=l2(1e-4), activation="relu", name="value_dense")(x)
value_output = Dense(
    1, kernel_regularizer=l2(1e-4), activation="tanh", name="value_output"
)(x)

model = Model(board_input, [policy_output, value_output], name="omega")
model.save("dlchess/models/omega_0", include_optimizer=False)
