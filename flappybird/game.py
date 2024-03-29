from tqdm import tqdm

from ple.games.flappybird import FlappyBird
from ple import PLE

import numpy as np


class FlappyGame:

    def __init__(self, return_rgb=False, display_screen=True, frame_skip=5, reward_clipping=False,
                 continuous_reward=False, leave_out_next_next=False, custom_reward=False):
        # Setup the environment
        self.game = PLE(FlappyBird(),
                        fps=30,
                        display_screen=display_screen,
                        frame_skip=frame_skip)
        self.valid_actions = self.game.getActionSet()
        self.return_rgb = return_rgb
        self.reward_clipping = reward_clipping
        self.continuous_reward = continuous_reward
        self.leave_out_next_next = leave_out_next_next
        self.custom_reward = custom_reward

        # Start the game and initialize values
        self.rewards = []
        self.game.init()

        # Now do one random action, such that the environment gets initialized properly
        self.game.act(np.random.choice(self.valid_actions, 1))

        if display_screen:  # verbose
            # Print Information About the Game
            print("FlappyBird Initialized!")
            print("\tValid Actions:", self.valid_actions)
            if self.return_rgb:
                print("\tReturning RBG numpy arrays of shape:", self.game.getScreenRGB().shape)
            else:
                print("\tReturning a dictionary with the features:")
                for key in self.game.getGameState().keys():
                    print("\t\t", key)

    def do_action(self, action):
        """
        Executes one action and returns the new state and the reward
        :param action:
        :return:
        """
        if action not in self.valid_actions:
            reward = self.game.act(self.valid_actions[action])
        else:
            reward = self.game.act(action)
        if self.reward_clipping:
            reward = 1 if reward > 0 else -1 if reward < 0 else 0
        elif self.continuous_reward:
            reward = 1 if reward >= 0 else 0
        self.rewards.append(reward)
        state = self.get_state()

        done = self.game.game_over()

        if self.custom_reward:
            player_y = state[0]
            pipe_top = state[3]
            pipe_bottom = state[4]

            reward = 200 - (player_y - (pipe_bottom + pipe_top)/2)
            reward /= 100

        return reward, state, done

    def get_state(self):
        if self.return_rgb:
            return self.game.getScreenRGB()
        else:
            features = list(self.game.getGameState().values())
            features = features[0:5]
            return features


    def reset(self):
        self.game.reset_game()
        # Do random action to initialize the new state
        self.game.act(np.random.choice(self.valid_actions, 1))
        return self.get_state()


def normalize_state(state):
    # return (state - state_min) / state_norm
    return (state - state_min[0:5]) / state_norm[0:5]


state_min = np.array([-14., -13., 1., 25., 125., 145., 25., 125.])
state_norm = \
    np.array([399., 10., 289., 192., 292., 433., 192., 292.]) - np.array([-14., -13.,  1., 25., 125., 145., 25., 125.])


def normalization_coefficients():
    flappyGame = FlappyGame(return_rgb=False, display_screen=False)
    n_test_frames = 100000
    initial_state = list(flappyGame.get_state().values())

    states = np.zeros(shape=(n_test_frames, len(initial_state)))

    for i in tqdm(range(n_test_frames)):
        random_action = np.random.choice(flappyGame.valid_actions, 1)
        _, state, done = flappyGame.do_action(random_action)
        states[i, :] = state
        if done:
            flappyGame.reset()

    #          player_y
    # 	       |     player_vel
    #          |     |     next_pipe_dist_to_player
    # 	       |     |     |     next_pipe_top_y
    # 	       |     |     |     |     next_pipe_bottom_y
    # 	       |     |     |     |     |      next_next_pipe_dist_to_player
    # 	       |     |     |     |     |      |    next_next_pipe_top_y
    # 	       |     |     |     |     |      |    |     next_next_pipe_bottom_y
    #          |     |     |     |     |      |    |     |
    # large: [399.,  10., 289., 192., 292., 433., 192., 292.]
    #        [399.,  10., 289., 192., 292., 433., 192., 292.]
    max_states = np.max(states, axis=0)
    print("Max States:", max_states)
    #          |     |     |     |     |      |    |     |
    # large: [-14., -13.,  1.,  25. , 125., 145.,  25., 125.]
    #        [-12., -13.,  1.,  26. , 126., 145.,  27., 127.]
    min_states = np.min(states, axis=0)
    print(min_states)

    # [167.,  -4., 169., 103., 203., 313., 127., 227.]
    median_states = np.median(states, axis=0)
    print("Median States:", median_states)

    return max_states, min_states, median_states


def _test():
    flappyGame = FlappyGame(return_rgb=False)
    n_test_frames = 10000000000
    for _ in range(n_test_frames):
        random_action = np.random.choice(flappyGame.valid_actions, 1)
        reward, state, done = flappyGame.do_action(random_action)
        print("Reward:", reward)
        print("Game State:", state)
        if done:
            print("died")
            flappyGame.reset()


if __name__ == "__main__":
    _test()
    # normalization_coefficients()
