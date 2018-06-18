import random
from collections import deque

import numpy as np
from keras import optimizers
import keras

from dqn.network import feature_q_network_conv
from flappybird.game import FlappyGame, normalize_state

def train_flappy_features():
    game = FlappyGame(return_rgb=True, display_screen=False, frame_skip=FRAMES_PER_ACTION, reward_clipping=True)
    state = np.array(normalize_state(game.get_state()))

    n_features = len(state)
    # This is actually 2 things, doing nothing or flying up. Depending on our implementation we could change it to 1?
    n_actions = len(game.valid_actions)

    epsilon = INITIAL_EPSILON
    replay_memory = deque()

    model = keras.models.load_model("dqn.h5")
    # model = feature_q_network_conv((CHANNELS, n_features), n_actions=n_actions)
    # model.compile(optimizer=optimizers.Nadam(lr=LEARNING_RATE), loss="MSE")

    s_t = np.stack((state, state, state, state), axis=0)
    s_t = np.expand_dims(s_t, axis=0)

    t = 0

    best_game = -1
    reward_game = 0
    last_100_games = []

    while True:
        loss = 0

        # Select an action
        if random.random() <= epsilon:
            action_index = np.random.randint(n_actions, dtype=int)
        else:
            Qs = model.predict(s_t)
            action_index = np.argmax(Qs, axis=-1)[0]
        action = game.valid_actions[action_index]

        # Execute the action and get it into the correct dims
        r_t, x_t1, done = game.do_action(action)
        x_t1 = np.array(normalize_state(x_t1))
        x_t1 = x_t1.reshape((1, 1, n_features))
        s_t1 = np.append(x_t1, s_t[:, :3, :], axis=1)

        reward_game += r_t

        # Store the experience in the replay memory
        replay_memory.append([s_t, action_index, r_t, s_t1, done])
        if len(replay_memory) > REPLAY_MEMORY:
            replay_memory.popleft()

        if t > WARM_UP:  # Only train when the warmup period is over
            batch = random.sample(replay_memory, BATCH_SIZE)

            state_t, action_t, reward_t, state_t1, terminal = zip(*batch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            targets[range(BATCH_SIZE), action_t] = reward_t + GAMMA * np.max(Q_sa, axis=1) * np.invert(terminal,
                                                                                                       dtype=int)

            loss += model.train_on_batch(state_t, targets)

        if epsilon > FINAL_EPSILON and t > WARM_UP:  # Lower epsilon
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / DECAY_EPSILON

        s_t = s_t1
        t += 1

        if t % 1000 == 0:
            model.save("dqn.h5", overwrite=True)
        if done:
            best_game = reward_game if reward_game > best_game else best_game
            last_100_games.append(reward_game)
            if len(last_100_games) > 100:
                del last_100_games[0]

            print(
                "Epoch: {:05d} \t Score: {:+04d} \t Avg@100: {:+04f} \t High Score: {:+04d} \t "
                "Loss: {:05f} \t Epsilon: {:04f}".format(
                    t, reward_game, np.mean(last_100_games), best_game, loss, epsilon)
            )
            reward_game = 0
            game.reset()


if __name__ == "__main__":
    GAMMA = 0.9
    WARM_UP = 3200
    FINAL_EPSILON = 0.0001
    INITIAL_EPSILON = 0.125
    DECAY_EPSILON = 125000.
    REPLAY_MEMORY = 50000
    BATCH_SIZE = 64
    FRAMES_PER_ACTION = 1
    LEARNING_RATE = 1e-6
    CHANNELS = 4

    train_flappy_features()
