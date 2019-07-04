from abc import ABCMeta, abstractmethod


class Agent(metaclass=ABCMeta):
    """Abstract Agent Class"""

    def __init__(self, id=None, name=None, alpha=None, training=None, policy=None):
        """
        :param name:
        :param alpha:
        :param training:
        :param policy:
        """
        self.id = id
        self.name = name
        self.alpha = alpha
        self.training = training
        self.policy = policy
        self.reward_history = []

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def observe(self, next_state):
        pass

    @abstractmethod
    def get_config(self):
        pass
