import gym
from gym import wrappers
import random
import numpy as np
from utils import rgb2gray, imresize, get_time

class Environment(object):
  def __init__(self, config):
    self.env = gym.make(config.env_name)

    if not config.is_train:
      self.gym_dir = '/tmp/%s-%s' % (config.env_name, get_time())
      self.env = gym.wrappers.Monitor(self.env, self.gym_dir)

    screen_width, screen_height, self.action_repeat, self.random_start = config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_width, screen_height)

    self._screen = None
    self.reward = 0
    self.terminal = False

    self.env.reset()

  def new_game(self):
    if self.terminal:
      self._screen = self.env.reset()
    
    self._random_step()
    self.render()
    return self.screen, 0, 0, self.terminal

  def new_random_game(self):
    self.new_game()
    for _ in xrange(random.randint(0, self.random_start - 1)):
      self._random_step()
    self.render()
    return self.screen, 0, 0, self.terminal

  def _step(self, action):
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  def _random_step(self):
    action = self.env.action_space.sample()
    self._step(action)

  @property
  def screen(self):
    return imresize(rgb2gray(self._screen)/255., self.dims)

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  def render(self):
    if self.display:
      self.env.render()

  def upload_game_session(self):
    self.env.close()
    gym.upload(self.gym_dir, api_key='sk_QPoIOdCKR5qkSsJ9UeG0dA')

class GymEnvironment(Environment):
  def __init__(self, config):
    super(GymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    cumulated_reward = 0

    for _ in xrange(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward

      if is_training and self.terminal:
        cumulated_reward -= 1

      if self.terminal:
        break

    self.reward = cumulated_reward

    self.render()
    return self.state
