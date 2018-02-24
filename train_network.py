"""
Trains The Neural Net.
"""

import argparse
import logging
import os
import random

import tensorflow as tf

from Model.input_fn import input_fn
from Model.utils import Params, set_logger, save_dict_to_json
from Model.model_fn import model_fn
from Model.training import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_dir', default='./Experiments/base_model',
    help="The experiment directory in which params.json\
    for the current experiment is defined."
)
parser.add_argument(
    '--data_dir', default='./Data/WHEAT/Binary/256x256_Binary',
    help="The Data directory."
)
parser.add_argument(
    '--restore_from', default=None,
    help="Optional, directory or file containing weights\
    to reload before training"
)

if __name__ == '__main__':
    tf.set_random_seed(753) # reproducibility
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert  os.path.isfile(json_path), "no config file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")
    # test_data_dir = os.path.join(data_dir, "test")

    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)
                       if f.endswith('.jpg')]
    eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)
                      if f.endswith('.jpg')]

    train_labels = [int(f.split('/')[-1][0]) for f in train_filenames]
    eval_labels = [int(f.split('/')[-1][0]) for f in eval_filenames]

    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

     # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_filenames, train_labels, params)
    eval_inputs = input_fn(False, eval_filenames, eval_labels, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(
        train_model_spec, eval_model_spec,
        args.model_dir, params, args.restore_from
    )
