import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim

from base import BaseModel
from replay_memory import ReplayMemory, History
from utils import get_time, save_pkl, load_pkl, linear, conv2d, clipped_error

class Agent(BaseModel):
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)
    self.epsilon = 0

    self.total_q = 0
    self.total_target_q = 0

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def train(self):
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []

    screen, reward, action, terminal = self.env.new_random_game()

    for _ in range(self.history_length):
      self.history.add(screen)

    tqdm_range = tqdm(range(start_step, self.max_step), initial=start_step)
    for self.step in tqdm_range:
      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      action = self.predict(self.history.get())
      
      screen, reward, terminal = self.env.act(action, is_training=True)
      
      self.observe(screen, reward, action, terminal)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
      total_reward += reward

      avg_reward = total_reward / (self.test_step or 1)
      avg_loss = self.total_loss / (self.update_count or 1)
      avg_q = self.total_q / (self.update_count or 1)

      try:
        max_ep_reward = np.max(ep_rewards)
        min_ep_reward = np.min(ep_rewards)
        avg_ep_reward = np.mean(ep_rewards)
      except: 
        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

      tqdm_range.set_postfix(avg_reward=avg_reward, avg_loss=avg_loss, avg_q=avg_q, num_game=num_game, \
          max_ep_reward=max_ep_reward, min_ep_reward=min_ep_reward, avg_ep_reward=avg_ep_reward, epsilon=self.epsilon)

      if self.step >= self.learn_start and self.step % self.test_step == 0:
        
        if max_avg_ep_reward * self.discount <= avg_ep_reward:
          self.step_assign_op.eval({ self.step_input: self.step + 1 })
          self.save_model(self.step + 1)

          max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

        num_game = 0
        total_reward = 0.
        self.total_loss = 0.
        self.total_q = 0.
        self.update_count = 0
        ep_reward = 0.
        ep_rewards = []
        actions = []

      if self.step >= self.learn_start and self.step % self.target_q_update_step == 0:
        self.inject_summary({
            'average.reward': avg_reward,
            'average.loss': avg_loss,
            'average.q': avg_q,
            'episode.max_reward': max_ep_reward,
            'episode.min_reward': min_ep_reward,
            'episode.avg_reward': avg_ep_reward,
            'episode.num_of_game': num_game,
            'episode.rewards': ep_rewards,
            'episode.actions': actions,
            'training.learning_rate': self.learning_rate_op.eval({ self.learning_rate_step: self.step }),
          }, self.step)

  def predict(self, s_t, test_ep=None):
    self.epsilon = test_ep or (self.ep_end + max(0., (self.ep_start - self.ep_end) * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < self.epsilon:
      action = random.randrange(self.env.action_size)
    else:
      actions = self.q_action.eval({
        self.s_t: [s_t],
        # self.train_length: 1,
        # self.images_batch_size: 1
      })
      action = actions[0]

    return action

  def error(self):
    return abs(self.total_target_q - self.total_q)

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal, self.error())

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()

      if self.step % self.target_q_update_step == 0:
        self.update_target_q_network()

  def q_learning_mini_batch(self):
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.memory.sample(self.error())

    pred_action = self.q_action.eval({
      self.s_t: s_t_plus_1
    })

    q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
      self.target_s_t: s_t_plus_1,
      self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)],
      #   # self.train_length: 4,
      #   # self.images_batch_size: 8
    })

    target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward

    _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
      # self.train_length: 4,
      # self.images_batch_size: 8 
    })

    self.writer.add_summary(summary_str, self.step)
    self.total_loss += loss
    self.total_q += q_t.mean()
    self.total_target_q += target_q_t.mean()
    self.update_count += 1

  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    self.train_length = tf.placeholder(dtype=tf.int32, name="train_length")
    self.images_batch_size = tf.placeholder(dtype=tf.int32, name="images_batches_size")

    with tf.variable_scope('prediction'):
      self.s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.history_length], name='s_t')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t, 32, [8, 8], [4, 4], initializer, activation_fn, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1, 64, [4, 4], [2, 2], initializer, activation_fn, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2, 64, [3, 3], [1, 1], initializer, activation_fn, name='l3')

      # self.l4, self.w['l4_w'], self.w['l4_b'] = conv2d(self.l3, 512, [7, 7], [1, 1], initializer, activation_fn, name='l4')

      ##############################################

      # rnn_cell_predict = tf.contrib.rnn.BasicLSTMCell(num_units=512, state_is_tuple=True)

      # conv_flat_predict = tf.reshape(slim.flatten(self.l4), [self.images_batch_size, self.train_length, 512])

      # state_in_predict = rnn_cell_predict.zero_state(self.images_batch_size, tf.float32)

      # # rnn_state_predict could be the point of convergence between the two kinds of networks
      # self.rnn_predict_input, self.rnn_state_predict = tf.nn.dynamic_rnn(inputs=conv_flat_predict, cell=rnn_cell_predict, dtype=tf.float32, initial_state=state_in_predict, scope='prediction_rnn')
      # self.rnn_predict_input = tf.reshape(self.rnn_predict_input, shape=[-1, 512])

      ##############################################

      # shape = self.rnn_predict_input.get_shape().as_list()
      # self.l4_flat = tf.reshape(self.rnn_predict_input, [-1, reduce(lambda x, y: x * y, shape[1:])])

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='value_hid')

      self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='adv_hid')

      self.value, self.w['val_w_out'], self.w['val_w_b'] = linear(self.value_hid, 1, name='value_out')

      self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = linear(self.adv_hid, self.env.action_size, name='adv_out')

      self.q = self.value + (self.advantage - tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))

      self.q_action = tf.argmax(self.q, dimension=1)

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0)
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))

      self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    with tf.variable_scope('target'):
      self.target_s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.history_length], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 32, [8, 8], [4, 4], initializer, activation_fn, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1, 64, [4, 4], [2, 2], initializer, activation_fn, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2, 64, [3, 3], [1, 1], initializer, activation_fn, name='target_l3')

      # self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = conv2d(self.target_l3, 512, [7, 7], [1, 1], initializer, activation_fn, name='target_l4')

      ##############################################

      # rnn_cell_target = tf.contrib.rnn.BasicLSTMCell(num_units=512,state_is_tuple=True)

      # state_in_target = rnn_cell_predict.zero_state(self.images_batch_size, tf.float32)

      # conv_flat_target = tf.reshape(slim.flatten(self.target_l4), [self.images_batch_size, self.train_length, 512])

      # rnn_state_predict could be the point of converge between the two kinds of networks
      # self.rnn_target, self.rnn_state_target = tf.nn.dynamic_rnn(inputs=conv_flat_target, cell=rnn_cell_target, dtype=tf.float32, initial_state=state_in_target, scope='target_rnn')
      # self.rnn_target = tf.reshape(self.rnn_target, shape=[-1, 512])

      ##############################################

      # shape = self.rnn_target.get_shape().as_list()
      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_value_hid')

      self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_adv_hid')

      self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = linear(self.t_value_hid, 1, name='target_value_out')

      self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

      # Average Dueling
      self.target_q = self.t_value + (self.t_advantage - tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted

      self.global_step = tf.Variable(0, trainable=False)

      
      # #In order to only propogate accurate gradients through the network, we will mask the first
      # #half of the losses for each trace as per Lample & Chatlot 2016
      # self.maskA = tf.zeros([self.images_batch_size, self.train_length / 2])
      # self.maskB = tf.ones([self.images_batch_size, self.train_length / 2])
      # self.mask = tf.concat([self.maskA, self.maskB], 1)
      # self.loss = tf.reduce_mean(clipped_error(self.delta * self.mask), name='loss')

      
      self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      
      exponential_decay = tf.train.exponential_decay(self.learning_rate, self.learning_rate_step, self.learning_rate_decay_step, self.learning_rate_decay, staircase=False)
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum, exponential_decay, name='learning_rate_exponential_decay')

      self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
      # self.optim = tf.train.AdamOptimizer(self.learning_rate_op, beta1=0.9, beta2=0.999, epsilon=1e-8, name='Adam_optim').minimize(self.loss)

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', 'episode.max_reward', 'episode.min_reward', 'episode.avg_reward', 'episode.num_of_game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.scalar("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

      self.writer = tf.summary.FileWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.global_variables_initializer().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    print(" [*] Updating target network...")
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    print(" [*] Updating summary...")
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })

    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, n_step=10000, n_episode=400, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    best_reward, best_idx = 0, 0

    self.env.env.reset()

    per_game_scores = []

    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for _ in xrange(9999999):

        action = self.predict(test_history.get(), test_ep)

        screen, reward, terminal = self.env.act(action, is_training=False)

        test_history.add(screen)

        current_reward += reward

        if terminal:
          per_game_scores.append(current_reward)
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print " [%d][%d] Best reward : %d, current reward %d, mean reward %d" % (idx, best_idx, best_reward, current_reward, np.mean(per_game_scores))

    # if best_reward > 13000:
    #   self.env.upload_game_session()