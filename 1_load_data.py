from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = base.load_csv_with_header(filename=IRIS_TRAINING,
                                         features_dtype=np.float64,
                                         target_dtype=np.int)
test_set = base.load_csv_with_header(filename=IRIS_TEST,
                                     features_dtype=np.float64,
                                     target_dtype=np.int)