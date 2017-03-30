import os
import random
import logging
import numpy as np

from utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, model_dir):
    self.model_dir = model_dir

    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0
    # self.worst_reward = None;

    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

  def add(self, screen, reward, action, terminal):
    assert screen.shape == self.dims
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):

    index = index % self.count

    if index >= self.history_length - 1:
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def sample(self):
    
    # if self.worst_reward == None:
    #   self.worst_reward = np.amin(self.rewards)
    #   print('Worst reward', self.worst_reward)

    # index = random.randint(self.history_length, self.count - self.batch_size)

    # for i in xrange(self.batch_size / 2):

    #   if self.rewards[index] <= self.worst_reward:
    #     break

    #   index = random.randint(self.batch_size, self.count - self.batch_size)

    # indexes = []
    # current_index = index - (self.batch_size / 2)

    # while len(indexes) < self.batch_size:
      
    #   self.prestates[len(indexes), ...] = self.getState(current_index - 1)
    #   self.poststates[len(indexes), ...] = self.getState(current_index)
    #   indexes.append(current_index)

    #   current_index += 1

    # actions = self.actions[indexes]
    # rewards = self.rewards[indexes]
    # terminals = self.terminals[indexes]

    # return np.transpose(self.prestates, (0, 2, 3, 1)), actions, rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals

    indexes = []
    while len(indexes) < self.batch_size:
      
      while True:
        
        index = random.randint(self.history_length, self.count - 1)
        
        if index >= self.current and index - self.history_length < self.current:
          continue
        
        if self.terminals[(index - self.history_length):index].any():
          continue
        break
      
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    return np.transpose(self.prestates, (0, 2, 3, 1)), actions, rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))


class History:
  def __init__(self, config):

    batch_size, history_length, screen_height, screen_width = config.batch_size, config.history_length, config.screen_height, config.screen_width

    self.history = np.zeros([history_length, screen_height, screen_width], dtype=np.float32)

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    return np.transpose(self.history, (1, 2, 0))



#### Prioritized experience replay

# class ReplayMemory:
#     e = 0.01
#     a = 0.6

#     def __init__(self, config, model_dir):
#       self.tree = SumTree(config.memory_size)
#       self.batch_size = config.batch_size
#       self.history_length = config.history_length
#       self.dims = (config.screen_height, config.screen_width)
#       self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
#       self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
#       self.count = 0

#     def _getPriority(self, error):
#       return (abs(error) + self.e) ** self.a

#     def add(self, screen, reward, action, terminal):
#       priority = self._getPriority(reward)
#       self.tree.add(priority, (screen, action, reward, terminal))
#       self.count = self.tree.total()

#     def sample(self):
#       segment = self.tree.total() / self.batch_size

#       actions = np.empty(self.batch_size, dtype = np.uint8)
#       rewards = np.empty(self.batch_size, dtype = np.uint8)
#       terminals = np.empty(self.batch_size, dtype = np.uint8)

#       for i in range(self.batch_size):
#         a = segment * i
#         b = segment * (i + 1)

#         s = random.uniform(a, b)
#         (index, priority, (screen, action, reward, terminal)) = self.tree.get(s)

#         actions[i] = action
#         rewards[i] = reward
#         terminals[i] = terminal

#         (_, _, (screen_previous, _, _, _)) = self.tree.get(s - 1)
#         (_, _, (screen_current, _, _, _)) = self.tree.get(s)

#         self.prestates[i] = screen_previous
#         self.poststates[i] = screen_current

#       return np.transpose(self.prestates, (0, 2, 3, 1)), actions, rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals

#     # def update(self, idx, error):
#     #     p = self._getPriority(error)
#     #     self.tree.update(idx, p)

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])    