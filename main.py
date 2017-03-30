import random
import tensorflow as tf

from dqn.agent import Agent
from dqn.environment import GymEnvironment
from config import get_config

flags = tf.app.flags

flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 54321, 'Value of random seed')

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

def main(_):

  with tf.Session() as sess:
    config = get_config(FLAGS) or FLAGS

    env = GymEnvironment(config)  

    agent = Agent(config, env, sess)

    if FLAGS.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
