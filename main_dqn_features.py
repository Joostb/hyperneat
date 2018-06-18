import numpy as np

from dqn.agent import DDQNAgent
from dqn.network import feature_q_network_dense
from flappybird.game import FlappyGame, normalize_state


def train_flappy_features():
    game = FlappyGame(return_rgb=False, display_screen=False, frame_skip=2, reward_clipping=True)
    s_t = np.array(normalize_state(game.get_state()))

    n_features = len(s_t)
    # This is actually 2 things, doing nothing or flying up. Depending on our implementation we could change it to 1?
    n_actions = len(game.valid_actions)

    agent = DDQNAgent((n_features,), n_actions, feature_q_network_dense)

    reward_100 = []
    best_game = -1

    for t in range(N_GAMES):
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
    log_file = open("log.csv", "w")
    log_file.write("game, score, epsilon, loss\n")
    N_GAMES = 50000
    WARM_UP = 1000
    BATCH_SIZE = 32

    train_flappy_features()
    log_file.close()
