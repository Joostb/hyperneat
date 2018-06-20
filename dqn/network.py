import keras.layers as L
from keras import Model
from keras.utils import plot_model


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


def feature_q_network_conv(input_shape=(4, 8), n_actions=2):
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


def feature_q_network_dense(input_shape=(8,), n_actions=2):
    input = L.Input(shape=input_shape, name="features")

    x = L.Dense(32, activation="relu", name="fc1")(input)
    x = L.BatchNormalization(name="bn_1")(x)

    x = L.Dense(64, activation="relu", name="fc2")(x)
    x = L.BatchNormalization(name="bn_2")(x)

    x = L.Dense(32, activation="linear", name="fc3")(x)
    x = L.BatchNormalization(name="bn3")(x)

    output = L.Dense(n_actions, name="Q_Layer")(x)

    model = Model(inputs=[input], outputs=[output])
    model.summary()

    return model


if __name__ == "__main__":
    # deep_q_network((4, 84, 84), n_actions=2)
    # feature_q_network_conv((4, 8), n_actions=2)
    model = feature_q_network_dense()
    plot_model(model, to_file="../results/feature_q_network.png", show_shapes=True, rankdir="LR")
