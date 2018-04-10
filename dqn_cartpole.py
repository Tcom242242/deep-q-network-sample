# https://github.com/openai/gym/wiki/CartPole-v0
import tensorflow as tf
import gym
from agents.memory import Memory
from agents.policy import EpsGreedyQPolicy
from agents.dqn import DQNAgent
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

env = gym.make('CartPole-v0')  # ゲームを指定して読み込む
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
actions = np.arange(nb_actions)
policy = EpsGreedyQPolicy(1.0, 0.999)
memory = Memory(limit=50000, maxlen=1)
obs = env.reset()
agent = DQNAgent(actions=actions, memory=memory, update_interval=500, train_interval=1, batch_size=32,
                 memory_interval=1, observation=obs, input_shape=[len(obs)], id=1, name=None, alpha=0.5,
                 training=True, policy=policy)
agent.compile()

result = []
for episode in range(500):  # 1000エピソード回す
    agent.reset()
    observation =  env.reset() # 環境の初期化
    # observation, _, _, _ = env.step(env.action_space.sample())
    observation = deepcopy(observation)
    agent.observe(observation)
    for t in range(250): # n回試行する
        # env.render() # 表示
        action = deepcopy(agent.act())
        observation, reward, done, info = env.step(action) #　アクションを実行した結果の状態、報酬、ゲームをクリアしたかどうか、その他の情報を返す
        observation = deepcopy(observation)
        agent.get_reward(reward, done)
        agent.observe(observation)
        if done:
            # if t < 195:
            #     reward = -1
            # else:
            #     reward = 1
            # agent.get_reward(reward, done)
            # agent.observe(observation)
            break
        # else:
        #     agent.get_reward(reward, done)
        #     agent.observe(observation)

    # test
    agent.training = False
    observation = env.reset() # 環境の初期化
    agent.observe(observation)
    for t in range(250):
        # env.render() # 表示
        action = agent.act()
        observation, reward, done, info = env.step(action)
        agent.observe(observation)
        # agent.get_reward(reward, done)
        if done:
            print("Episode {}, maintain {} timesteps".format(episode, t))
            result.append(t)
            break
    agent.training = True

x = np.arange(len(result))
plt.ylabel("time")
plt.xlabel("episode")
plt.ylim((0, 200))
plt.plot(x, result)
plt.savefig("result.png")

