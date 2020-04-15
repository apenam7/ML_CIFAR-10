# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:39:45 2020

@author: apena
"""
# import sys
# import numpy as np
# import tensorflow as tf
# from datetime import datetime

# device_name = 'gpu'  # Choose device from cmd line. Options: gpu or cpu
# shape = (1500, 1500)
# if device_name == "gpu":
#     device_name = "/gpu:0"
# else:
#     device_name = "/cpu:0"

# with tf.device(device_name):
#     random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
#     dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
#     sum_operation = tf.reduce_sum(dot_operation)


# startTime = datetime.now()
# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as session:
#         result = session.run(sum_operation)
#         print(result)

# # It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
# print("\n" * 5)
# print("Shape:", shape, "Device:", device_name)
# print("Time taken:", datetime.now() - startTime)

# print("\n" * 5)
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)