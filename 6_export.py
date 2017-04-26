from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

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

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
def predict_fn():
  new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
  return {"flower_features": new_samples}

Ys = classifier.predict(input_fn=predict_fn)
for y in Ys:
  print('Predictions: {}'.format(str(y)))

# export the model
def serving_input_fn():
  feature_placeholders = {
    "flower_features": tf.placeholder(tf.float32, shape=[None, 4])
  }
  # DNNClassifier expects rank 2 Tensors, but inputs should be
  # rank 1, so that we can provide scalars to the server
  features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
  }

  return input_fn_utils.InputFnOps(
      features=features, # input into graph
      labels=None,
      default_inputs=feature_placeholders # tensor input converted from request
  )

export_folder = classifier.export_savedmodel(
    export_dir_base = model_dir + '/export',
    input_fn=serving_input_fn
)
print('model exported successfully to {}'.format(export_folder))

