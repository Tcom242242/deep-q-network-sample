import tensorflow as tf
from agents.agent import Agent
import numpy as np
import agents.network as net

class DQNAgent(Agent):
    """
        keras-rlのコードを参考にしたDQNエージェント
    """
    def __init__(self, gamma=0.99, alpha_decay_rate=0.999, actions=None, memory=None, memory_interval=1,train_interval=1, 
                 batch_size=32, update_interval=10, nb_steps_warmup=100, observation=None,
                 input_shape=None, 
                 **kwargs):

        super().__init__(**kwargs)
        self.actions = actions
        self.gamma = gamma
        self.state = observation
        self.alpha_decay_rate = alpha_decay_rate
        self.recent_observation = observation
        self.update_interval = update_interval
        self.memory = memory
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.recent_action_id = 0
        self.nb_steps_warmup = nb_steps_warmup
        self.sess = tf.InteractiveSession()
        self.model_inputs, self.model_outputs, self.model_max_outputs, self.model = net.build_model(input_shape,len(self.actions))
        self.target_model_inputs, self.target_model_outputs, self.target_model_max_outputs, self.target_model= net.build_model(input_shape, len(self.actions))
        target_model_weights = self.target_model.trainable_weights
        model_weights = self.model.trainable_weights
        self.update_target_model = [target_model_weights[i].assign(model_weights[i]) for i in range(len(target_model_weights))]
        self.train_interval = train_interval
        self.step = 0

    def compile(self):
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None], name="target_q")
        self.inputs= tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        actions_one_hot = tf.one_hot(indices=self.inputs, depth=len(self.actions), on_value=1.0, off_value=0.0, name="action_one_hot")

        self.pred_q = tf.reduce_sum(tf.multiply(self.model_outputs,actions_one_hot), reduction_indices=1, name="q_acted")
        self.delta = tf.abs(self.targets - self.pred_q)

        # huber loss
        self.clipped_error = tf.where(self.delta < 1.0,
                                      0.5 * tf.square(self.delta),
                                      self.delta - 0.5, name="clipped_error")
        self.loss = tf.reduce_mean(self.clipped_error, name="loss")

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train = optimizer.minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())

    def update_target_model_hard(self):
        """ copy q-network to target network """
        self.sess.run(self.update_target_model)

    def train_on_batch(self, state_batch, action_batch, targets):
        self.sess.run(self.train, feed_dict={self.model_inputs:state_batch, self.inputs:action_batch, self.targets:targets})

    def predict_on_batch(self, state1_batch):
        q_values = self.sess.run(self.target_model_max_outputs, feed_dict={self.target_model_inputs:state1_batch})
        return q_values

    def compute_q_values(self, state):
        q_values = self.sess.run(self.model_outputs, feed_dict={self.model_inputs:[state]})
        return q_values[0]

    def get_reward(self, reward, terminal):
        self.reward_history.append(reward)
        if self.training:
            self._update_q_value(reward, terminal)

        self.decay_alpha()
        self.policy.decay_eps_rate()
        self.step += 1

    def _update_q_value(self, reward, terminal):
        self.backward(reward, terminal)

    def backward(self, reward, terminal):
        if self.step % self.memory_interval == 0:
            """ store experience """
            self.memory.append(self.recent_observation, self.recent_action_id, reward, terminal=terminal, training=self.training)

        if (self.step > self.nb_steps_warmup) and (self.step % self.train_interval == 0):
            experiences = self.memory.sample(self.batch_size)

            state0_batch = []
            reward_batch = []
            action_batch = []
            state1_batch = []
            terminal_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal_batch.append(0. if e.terminal else 1.)

            reward_batch = np.array(reward_batch)
            target_q_values = np.array(self.predict_on_batch(state1_batch))   # compute maxQ'(s')
            discounted_reward_batch = (self.gamma * target_q_values)
            discounted_reward_batch *= terminal_batch
            targets = reward_batch + discounted_reward_batch    # target = r + γ maxQ'(s')
            self.train_on_batch(state0_batch, action_batch, targets)

        if self.step % self.update_interval == 0:
            """ update target network """
            self.update_target_model_hard()

    def act(self):
        action_id = self.forward()
        action = self.actions[action_id]
        return action

    def forward(self):
        state = self.recent_observation
        q_values = self.compute_q_values(state)
        if self.training:
            action_id = self.policy.select_action(q_values=q_values)
        else:
            action_id = self.policy.select_greedy_action(q_values=q_values)

        self.recent_action_id = action_id
        return action_id

    def observe(self, next_state):
        self.recent_observation = next_state

    def decay_alpha(self):
        self.alpha = self.alpha*self.alpha_decay_rate

    def reset(self):
        self.recent_observation = None
        self.recent_action_id = None

    def get_data(self):
        result = {}
        result["alpha"] = self.alpha
        result["gamma"] = self.gamma
        result["epsilon"] = self.policy.eps
        result["reward_history"] = self.reward_history
        return result
