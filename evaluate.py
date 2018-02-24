"""Evaluate the Neural Net using the tests set """

import argparse
import logging
import os

import tensorflow as tf

from Model.input_fn import input_fn
from Model.model_fn import model_fn
from Model.evaluation import evaluate
from Model.utils import Params, set_logger

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', default='./Experiments/test',
    help="The experiment directory in which params.json\
    for the current experiment is defined."
)
parser.add_argument(
    '--data_dir', default='data/WHEAT/Binary/',
    help="Directory with the dataset"
)
parser.add_argument(
    '--restore_from', default='best_weights',
    help="Subdirectory of model dir or file containing the weights"
)

if __name__ == '__main__':
    tf.set_random_seed(753)
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert  os.path.isfile(json_path), "no config file found at {}".format(json_path)
    params = Params(json_path)

    set_logger(os.path.join(args.model_dir, 'evaluate.log'))
    logging.info("Creating the dataset...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test")

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames
                      if f.endswith('.jpg')]
    test_labels = [int(f.split('/')[-1][0]) for f in test_filenames]
    params.eval_size = len(test_filenames)

    # create the iterator over the dataset
    test_inputs = input_fn(False, test_filenames, test_labels, params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, args.restore_from)
