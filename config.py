class AgentConfig(object):
  scale = 10000
  display = False

  max_step = 5000 * scale
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.9

  target_q_update_step = 1 * scale
  learning_rate = 0.0001
  learning_rate_minimum = 0.0001
  learning_rate_decay = 0.99
  learning_rate_decay_step = 10 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size / 5

  history_length = 2
  train_frequency = 2
  action_repeat = 2
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  _test_step = 5 * scale
  _save_step = _test_step * 10

  env_name = ''

  screen_width  = 84
  screen_height = 84
  max_reward = 1.
  min_reward = -1.

  backend = 'tf'
  env_type = 'detail'

  is_train = False

def get_config(FLAGS):
  
  config = AgentConfig

  for k, v in FLAGS.__dict__['__flags'].items():

    if hasattr(config, k):
      setattr(config, k, v)

  return config
