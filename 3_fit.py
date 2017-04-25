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

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("flower_features", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
model_dir="/tmp/iris_model"
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir=model_dir)

# Fit model.
def input_fn(dataset): 
  def _fn():
    features = {"flower_features": tf.constant(dataset.data)}
    label = tf.constant(dataset.target)
    return features, label
  return _fn

classifier.fit(input_fn=input_fn(training_set),
               steps=1000)
print('fit done')