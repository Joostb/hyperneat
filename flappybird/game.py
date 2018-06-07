from ple.games.flappybird import FlappyBird
from ple import PLE

import numpy as np


class FlappyGame:

    def __init__(self, return_rgb=False, display_screen=True, frame_skip=5):
        # Setup the environment
        self.game = PLE(FlappyBird(),
                        fps=30,
                        display_screen=display_screen,
                        frame_skip=frame_skip)
        self.valid_actions = self.game.getActionSet()
        self.return_rgb = return_rgb

        # Start the game and initialize values
        self.rewards = []
        self.game.init()

        # Now do one random action, such that the environment gets initialized properly
        self.game.act(np.random.choice(self.valid_actions, 1))

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
        reward = self.game.act(action)
        self.rewards.append(reward)
        state = self.get_state()
        done = self.game.game_over()

        return reward, state, done

    def get_state(self):
        return self.game.getScreenRGB() if self.return_rgb else self.game.getGameState()

    def reset(self):
        self.game.reset_game()


def _test():
    flappyGame = FlappyGame(return_rgb=False)
    n_test_frames = 1000
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
