""""Performs Hyperparameter search"""

import argparse
import os
from subprocess import check_call
import sys

from Model.utils import Params

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument(
    '--parent_dir', default='experiments/learning_rate',
    help="Directory containing params.json"
)
parser.add_argument(
    '--data_dir', default='./Data/WHEAT/Binary/',
    help="The Data directory."
)
parser.add_argument(
    '--param_to_tune', default='lr',
    help="The parameter to tune, example: --param_to_tune=lr"
)
parser.add_argument(
    '--lr_num_times', default=4,
    help="Number of times to sample the learning rate."
)

def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters
    in parent_dir/job_name

    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train_network.py --model_dir {model_dir} --data_dir {data_dir}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)

if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No config file found at {}".format(json_path)
    params = Params(json_path)
    lr_num_times = int(args.lr_num_times)

    # Perform hypersearch over one parameter
    # random search over [10**-4 to 10**0]
    learning_rates = [10**(-4*np.random.rand()) for i in xrange(lr_num_times)]

    for learning_rate in learning_rates:
        params.learning_rate = learning_rate
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
