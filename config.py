class AgentConfig(object):
  scale = 10000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 1 * scale
  learning_rate = 0.001
  learning_rate_minimum = 0.00001
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size / 2

  history_length = 4
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10

  env_name = 'Breakout-v0'

  screen_width  = 84
  screen_height = 84
  max_reward = 1.
  min_reward = -1.

  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

  model = ''

def get_config(FLAGS):
  
  config = AgentConfig

  for k, v in FLAGS.__dict__['__flags'].items():

    if hasattr(config, k):
      setattr(config, k, v)

  return config
