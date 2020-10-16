import os
import argparse
import xnap.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--mode', default=0, type=int)
    """ There are three modes
        0 = train models
        1 = explain random process instance
        2 = explain and manipulate 
    """
    parser.add_argument('--rand_lower_bound', default=5, type=int)
    parser.add_argument('--rand_upper_bound', default=5, type=int)
    parser.add_argument('--task', default="nap")
    parser.add_argument('--data_set', default="helpdesk_sample.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--model_dir', default="nap/models/")
    parser.add_argument('--result_dir', default="./results/")

    # pm4py
    parser.add_argument('--case_id_key', default="case", type=str)
    parser.add_argument('--activity_key', default="event", type=str)

    # Parameters for deep neural network
    parser.add_argument('--dnn_num_epochs', default=1, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--dim', default=0, type=int)

    # Parameters for validation
    parser.add_argument('--num_folds', default=1, type=int)
    parser.add_argument('--cross_validation', default=False, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.8, type=float)
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # pre-processing
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)  # onehot or hash for numerical attributes
    parser.add_argument('--encoding_cat', default="onehot", type=str)  # onehot or hash for categorical attributes
    parser.add_argument('--num_hash_output', default=10, type=int)

    # Parameters for gpu processing
    parser.add_argument('--gpu_ratio', default=0.2, type=float)
    parser.add_argument('--cpu_num', default=1, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
