# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""A demo for on-device backprop (transfer learning) of a classification model.

This demo runs a similar task as described in TF Poets tutorial, except that
learning happens on-device.
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

Here are the steps:
  1) mkdir -p /tmp/retrain/

  2) curl http://download.tensorflow.org/example_images/flower_photos.tgz \
       | tar xz -C /tmp/retrain

  3) Start training:

      python3 backprop_last_layer.py \
      --data_dir /tmp/retrain/flower_photos \
      --embedding_extractor_path \
      models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite

     Weights for retrained last layer will be saved to /tmp/retrain/output by
     default.

  4) Run an inference with the new model:

      python3 classify_image.py \
      --model /tmp/retrain/output/retrained_model_edgetpu.tflite \
      --label /tmp/retrain/output/label_map.txt
      --image images/sunflower.bmp

For more information, see
https://coral.ai/docs/edgetpu/retrain-classification-ondevice-backprop/
"""

import argparse
import contextlib
import os
import sys
import time

from edgetpu.basic import basic_engine
from edgetpu.learn.backprop.softmax_regression import SoftmaxRegression
import numpy as np
from PIL import Image


@contextlib.contextmanager
def test_image(path):
  """Returns opened test image."""
  with open(path, 'rb') as f:
    with Image.open(f) as image:
      yield image


def save_label_map(label_map, out_path):
  """Saves label map to a file."""
  with open(out_path, 'w') as f:
    for key, val in label_map.items():
      f.write('%s %s\n' % (key, val))


def get_image_paths(data_dir):
  """Walks through data_dir and returns list of image paths and label map.

  Args:
    data_dir: string, path to data directory. It assumes data directory is
      organized as,
          - [CLASS_NAME_0]
            -- image_class_0_a.jpg
            -- image_class_0_b.jpg
            -- ...
          - [CLASS_NAME_1]
            -- image_class_1_a.jpg
            -- ...
  Returns:
    A tuple of (image_paths, labels, label_map)
    image_paths: list of string, represents image paths
    labels: list of int, represents labels
    label_map: a dictionary (int -> string), e.g., 0->class0, 1->class1, etc.
  """
  classes = None
  image_paths = []
  labels = []

  class_idx = 0
  for root, dirs, files in os.walk(data_dir):
    if root == data_dir:
      # Each sub-directory in `data_dir`
      classes = dirs
    else:
      # Read each sub-directory
      assert classes[class_idx] in root
      print('Reading dir: %s, which has %d images' % (root, len(files)))
      for img_name in files:
        image_paths.append(os.path.join(root, img_name))
        labels.append(class_idx)
      class_idx += 1

  return image_paths, labels, dict(zip(range(class_idx), classes))


def shuffle_and_split(image_paths, labels, val_percent=0.1, test_percent=0.1):
  """Shuffles and splits data into train, validation, and test sets.

  Args:
    image_paths: list of string, of dim num_data
    labels: list of int of length num_data
    val_percent: validation data set percentage.
    test_percent: test data set percentage.

  Returns:
    Two dictionaries (train_and_val_dataset, test_dataset).
    train_and_val_dataset has the following fields.
      'data_train': data_train
      'labels_train': labels_train
      'data_val': data_val
      'labels_val': labels_val
    test_dataset has the following fields.
      'data_test': data_test
      'labels_test': labels_test
  """
  image_paths = np.array(image_paths)
  labels = np.array(labels)
  perm = np.random.permutation(image_paths.shape[0])
  image_paths = image_paths[perm]
  labels = labels[perm]

  num_total = image_paths.shape[0]
  num_val = int(num_total * val_percent)
  num_test = int(num_total * test_percent)
  num_train = num_total - num_val - num_test

  train_and_val_dataset = {}
  train_and_val_dataset['data_train'] = image_paths[0:num_train]
  train_and_val_dataset['labels_train'] = labels[0:num_train]
  train_and_val_dataset['data_val'] = image_paths[num_train:num_train + num_val]
  train_and_val_dataset['labels_val'] = labels[num_train:num_train + num_val]
  test_dataset = {}
  test_dataset['data_test'] = image_paths[num_train + num_val:]
  test_dataset['labels_test'] = labels[num_train + num_val:]
  return train_and_val_dataset, test_dataset


def extract_embeddings(image_paths, engine):
  """Uses model to process images as embeddings.

  Reads image, resizes and feeds to model to get feature embeddings. Original
  image is discarded to keep maximum memory consumption low.

  Args:
    image_paths: ndarray, represents a list of image paths.
    engine: BasicEngine, wraps embedding extractor model.

  Returns:
    ndarray of length image_paths.shape[0] of embeddings.
  """
  _, input_height, input_width, _ = engine.get_input_tensor_shape()
  assert engine.get_num_of_output_tensors() == 1
  feature_dim = engine.get_output_tensor_size(0)

  embeddings = np.empty((len(image_paths), feature_dim), dtype=np.float32)
  for idx, path in enumerate(image_paths):
    with test_image(path) as img:
      img = img.resize((input_width, input_height), Image.NEAREST)
      _, embeddings[idx, :] = engine.run_inference(np.asarray(img).flatten())

  return embeddings


def train(model_path, data_dir, output_dir):
  """Trains a softmax regression model given data and embedding extractor.

  Args:
    model_path: string, path to embedding extractor.
    data_dir: string, directory that contains training data.
    output_dir: string, directory to save retrained tflite model and label map.
  """
  t0 = time.perf_counter()
  image_paths, labels, label_map = get_image_paths(data_dir)
  train_and_val_dataset, test_dataset = shuffle_and_split(image_paths, labels)
  # Initializes basic engine model here to avoid repeatedly initialization,
  # which is time consuming.
  engine = basic_engine.BasicEngine(model_path)
  print('Extract embeddings for data_train')
  train_and_val_dataset['data_train'] = extract_embeddings(
      train_and_val_dataset['data_train'], engine)
  print('Extract embeddings for data_val')
  train_and_val_dataset['data_val'] = extract_embeddings(
      train_and_val_dataset['data_val'], engine)
  t1 = time.perf_counter()
  print('Data preprocessing takes %.2f seconds' % (t1 - t0))

  # Construct model and start training
  weight_scale = 5e-2
  reg = 0.0
  feature_dim = train_and_val_dataset['data_train'].shape[1]
  num_classes = np.max(train_and_val_dataset['labels_train']) + 1
  model = SoftmaxRegression(
      feature_dim, num_classes, weight_scale=weight_scale, reg=reg)

  learning_rate = 1e-2
  batch_size = 100
  num_iter = 500
  model.train_with_sgd(
      train_and_val_dataset, num_iter, learning_rate, batch_size=batch_size)
  t2 = time.perf_counter()
  print('Training takes %.2f seconds' % (t2 - t1))

  # Append learned weights to input model and save as tflite format.
  out_model_path = os.path.join(output_dir, 'retrained_model_edgetpu.tflite')
  model.save_as_tflite_model(model_path, out_model_path)
  print('Model %s saved.' % out_model_path)
  label_map_path = os.path.join(output_dir, 'label_map.txt')
  save_label_map(label_map, label_map_path)
  print('Label map %s saved.' % label_map_path)
  t3 = time.perf_counter()
  print('Saving retrained model and label map takes %.2f seconds' % (t3 - t2))

  retrained_engine = basic_engine.BasicEngine(out_model_path)
  test_embeddings = extract_embeddings(test_dataset['data_test'],
                                       retrained_engine)
  saved_model_acc = np.mean(
      np.argmax(test_embeddings, axis=1) == test_dataset['labels_test'])
  print('Saved tflite model test accuracy: %.2f%%' % (saved_model_acc * 100))
  t4 = time.perf_counter()
  print('Checking test accuracy takes %.2f seconds' % (t4 - t3))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--embedding_extractor_path',
      required=True,
      help='Path to embedding extractor tflite model.')
  parser.add_argument('--data_dir', required=True, help='Directory to data.')
  parser.add_argument(
      '--output_dir',
      default='/tmp/retrain/output',
      help='Path to directory to save retrained model and label map.')
  args = parser.parse_args()

  if not os.path.exists(args.data_dir):
    sys.exit('%s does not exist!' % args.data_dir)

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  train(args.embedding_extractor_path, args.data_dir, args.output_dir)


if __name__ == '__main__':
  main()
