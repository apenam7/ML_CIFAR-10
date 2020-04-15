# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:47:21 2020

@author: apena
"""

import tensorflow as tf

# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert(tf.test.is_gpu_available())

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False) # Start with XLA disabled.

def load_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 256
  x_test = x_test.astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = load_data()