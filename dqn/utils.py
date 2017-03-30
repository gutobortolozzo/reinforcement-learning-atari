import time
import cPickle
import numpy as np
import tensorflow as tf

import cv2
imresize = cv2.resize

def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print(" [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print(" [*] save %s" % path)

@timeit
def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print(" [*] load %s" % path)
    return obj

@timeit
def save_npy(obj, path):
  np.save(path, obj)
  print(" [*] save %s" % path)

@timeit
def load_npy(path):
  obj = np.load(path)
  print(" [*] load %s" % path)
  return obj

def clipped_error(x):
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x, output_dim, kernel_size, stride, initializer=tf.contrib.layers.xavier_initializer(), activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):
  with tf.variable_scope(name):
    stride = [1, stride[0], stride[1], 1]
    kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]  

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format='NHWC')

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(conv, b, 'NHWC')

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
      return activation_fn(out), w, b
    else:
      return out, w, b
