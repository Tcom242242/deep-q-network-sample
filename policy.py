from abc import ABCMeta, abstractmethod
import numpy as np
import ipdb

class Policy(metaclass=ABCMeta):

    @abstractmethod
    def select_action(self, **kwargs):
        pass

    def get_config(self):
        return {}


class EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1, eps_decay_rate=0.99):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.eps_decay_rate = eps_decay_rate

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            # action = np.random.random_integers(0, nb_actions-1)
            action = np.argmax(q_values)


        return action

    def decay_eps_rate(self):
        self.eps = self.eps*self.eps_decay_rate
        if self.eps < 0.01:
            self.eps = 0.01


    def select_greedy_action(self, q_values):
        assert q_values.ndim == 1
        action = np.argmax(q_values)

        return action

    def get_config(self):
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config
