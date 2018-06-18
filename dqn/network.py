import keras.layers as L
from keras import Model


def deep_q_network(input_shape, n_actions=2):
    input = L.Input(shape=input_shape)

    x = L.Conv2D(32, kernel_size=8, strides=4, padding="same", activation="relu", name="conv_1")(input)
    x = L.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="relu", name="conv_2")(x)
    x = L.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu", name="conv_3")(x)

    x = L.Flatten()(x)

    x = L.Dense(512, activation="relu", name="features")(x)
    qs = L.Dense(n_actions, name="q-values")(x)

    model = Model(inputs=[input], outputs=qs)
    model.summary()

    return model


def feature_q_network(input_shape=(4, 8), n_actions=2):
    input = L.Input(shape=input_shape)

    x = L.Conv1D(32, kernel_size=2, padding="valid", activation="relu")(input)
    x = L.Conv1D(64, kernel_size=2, padding="valid", activation="relu")(x)
    x = L.Conv1D(128, kernel_size=2, padding="valid", activation="relu")(x)

    x = L.Flatten()(x)

    x = L.Dense(512, activation="relu")(x)

    output = L.Dense(n_actions)(x)

    model = Model(inputs=[input], outputs=[output])
    model.summary()

    return model


if __name__ == "__main__":
    deep_q_network((4, 84, 84), n_actions=2)
    feature_q_network((4, 8), n_actions=2)
