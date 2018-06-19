import random
from collections import deque

from keras import models

import keras.backend as K
import numpy as np
from keras import optimizers


class DDQNAgent:
    def __init__(self, state_dim, n_actions, model, start_epsilon=1.0):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.replay_memory = deque(maxlen=50000)
        self.gamma = 0.95  # discount rate
        self.epsilon = start_epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 1e-3
        self.model = self._build_model(model)
        self.target_model = self._build_model(model)
        self.update_target_model()
        self.loss = 0
        self.n_batches = 0

    @staticmethod
    def _huber_loss(target, prediction):
        """
        sqrt(1+error^2)-1
        """
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def _build_model(self, model):
        """
        Setup the model with the Huber Loss
        :return:
        """
        model = model(self.state_dim, n_actions=self.n_actions)
        model.compile(loss=self._huber_loss,
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Set weights of target model to the training model
        """
        self.target_model.set_weights(self.model.get_weights())

    def store(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        """
        self.replay_memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Return index of the best action given a state
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        Generate a batch and train on it
        """
        batch = random.sample(self.replay_memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.loss += self.model.train_on_batch(state, target)
            self.n_batches += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_avg_loss(self):
        """
        Get the avg loss since last calling this function
        """
        if self.n_batches > 0:
            avg_loss = self.loss / self.n_batches
            self.loss = 0
            self.n_batches = 0
            return avg_loss
        else:
            return 0

    def load(self, name):
        self.model = models.load_model(name, custom_objects={"_huber_loss": self._huber_loss})

    def save(self, name):
        self.model.save(name)
