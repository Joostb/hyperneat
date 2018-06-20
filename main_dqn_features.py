import numpy as np

from dqn.agent import DDQNAgent
from dqn.network import feature_q_network_dense
from flappybird.game import FlappyGame, normalize_state

import argparse


def play_flappy():
    game = FlappyGame(return_rgb=False, display_screen=True, frame_skip=2, reward_clipping=True)
    s_t = np.array(normalize_state(game.get_state()))

    n_features = len(s_t)
    # This is actually 2 things, doing nothing or flying up. Depending on our implementation we could change it to 1?
    n_actions = len(game.valid_actions)

    agent = DDQNAgent((n_features,), n_actions, feature_q_network_dense, start_epsilon=0.)
    agent.load("flappy_dqn.h5")
    agent.update_target_model()

    rewards = []

    for t in range(N_GAMES):
        s_t = normalize_state(game.reset())
        s_t = np.expand_dims(s_t, axis=0)
        total_reward = 0
        done = False

        while not done:
            action_index = agent.act(s_t)
            reward, s_t, done = game.do_action(game.valid_actions[action_index])

            total_reward += reward

            s_t = np.expand_dims(normalize_state(s_t), axis=0)
            if done:
                rewards.append(total_reward)
                print(
                    "Game: {:05d} \t Score: {:+03f} \t Avg Score: {:+04f} \t High Score: {:+04f}".format(
                        t, total_reward, np.mean(rewards), np.max(rewards))
                )
                log_file.write("{}, {}".format(t, total_reward))


def train_flappy_features(begin_epoch=0, begin_epsilon=1.0):
    game = FlappyGame(return_rgb=False, display_screen=False, frame_skip=2, reward_clipping=True, leave_out_next_next=True)
    s_t = np.array(normalize_state(game.get_state()))

    n_features = len(s_t)
    # This is actually 2 things, doing nothing or flying up. Depending on our implementation we could change it to 1?
    n_actions = len(game.valid_actions)

    agent = DDQNAgent((n_features,), n_actions, feature_q_network_dense, start_epsilon=begin_epsilon)

    reward_100 = []
    best_game = -1

    for t in range(begin_epoch, N_GAMES+begin_epoch):
        s_t = normalize_state(game.reset())
        s_t = np.expand_dims(s_t, axis=0)
        total_reward = 0
        done = False

        while not done:
            action_index = agent.act(s_t)
            reward, s_t1, done = game.do_action(game.valid_actions[action_index])

            total_reward += reward

            s_t1 = np.expand_dims(normalize_state(s_t1), axis=0)
            agent.store(s_t, action_index, reward, s_t1, done)
            s_t = s_t1
            if done:
                agent.update_target_model()

                reward_100.append(total_reward)
                if len(reward_100) > 100:
                    del reward_100[0]
                best_game = total_reward if total_reward > best_game else best_game
                game_loss = agent.get_avg_loss()
                print(
                    "Game: {:05d} \t Score: {:+03d} \t Avg@100: {:+04f} \t High Score: {:+04d} \t "
                    "Epsilon: {:04f} \t Game Loss: {:05f}".format(
                        t, total_reward, np.mean(reward_100), best_game, agent.epsilon, game_loss)
                )
                log_file.write("{}, {}, {}, {}\n".format(t, total_reward, agent.epsilon, game_loss))
                log_file.flush()
            if len(agent.replay_memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
            if t % 10 == 0:
                agent.save("flappy_dqn.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN Flappy Bird")
    parser.add_argument("--train", dest="train", action="store_true", default=False)
    parser.add_argument("--resume", dest="resume", action="store_true", default=False)
    parser.add_argument("--n-games", dest="n_games", nargs="?", default=500, type=int)
    args = parser.parse_args()

    N_GAMES = args.n_games

    if args.train:
        if args.resume:
            prev_log = np.genfromtxt("log.csv", dtype=float, delimiter=',', skip_header=True)
            begin_epoch = prev_log[-1, 0]
            begin_epsilon = prev_log[-1, 2]
            log_file = open("log.csv", "a")
        else:
            log_file = open("log.csv", "w")
            log_file.write("game, score, epsilon, loss\n")
            begin_epoch = 0
            begin_epsilon = 0.2

        WARM_UP = 1000
        BATCH_SIZE = 32

        train_flappy_features(begin_epoch=begin_epoch, begin_epsilon=begin_epsilon)
        log_file.close()
    else:
        log_file = open("run.csv", "w")
        log_file.write("game, score\n")
        play_flappy()
        log_file.close()
