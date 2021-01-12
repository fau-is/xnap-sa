import os
import argparse
import xnap.utils as utils


def load():
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--mode', default=0, type=int)
    """ There are three modes
        0 = train and test model
        1 = explain prediction for random process instance
        2 = evaluate explanations for predictions (test set) 
    """
    # Mode 1 + 2
    parser.add_argument('--xai', default="lime", type=str)  # lrp, lime, shap
    parser.add_argument('--lime_num_samples', default=10, type=int)  # 5000 (default)
    parser.add_argument('--shap_num_samples', default=10, type=int)  # 100 (good estimate), 1000 (very good estimate)
    # Mode 1
    parser.add_argument('--rand_lower_bound', default=6, type=int)
    parser.add_argument('--rand_upper_bound', default=6, type=int)
    # Mode 2
    parser.add_argument('--removed_events_num', default=1, type=int)
    parser.add_argument('--removed_events_relevance', default="highest", type=str)  # lowest, highest
    parser.add_argument('--eager_execution', default=False, type=utils.str2bool)  # to avoid retracing with tf for lrp
    # Classifier
    #   LSTM -> Bi-directional long short-term neural network
    #   RF  -> Random Forest
    #   DT  -> Decision Tree
    parser.add_argument('--classifier', default="LSTM", type=str)  # LSTM, RF, DT

    # Parameters for deep neural network
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--dim', default=0, type=int)

    # Directories
    parser.add_argument('--task', default="nap")
    parser.add_argument('--data_set', default="helpdesk_raw.csv")
    parser.add_argument('--data_dir', default="./data/")
    parser.add_argument('--model_dir', default="nap/models/")
    parser.add_argument('--result_dir', default="./results/")

    # Parameters for validation
    parser.add_argument('--seed', default=True, type=utils.str2bool)
    parser.add_argument('--seed_val', default=1377, type=int)
    parser.add_argument('--shuffle', default=True, type=int)
    parser.add_argument('--split_rate_train', default=0.99, type=float)
    parser.add_argument('--val_split', default=0.1, type=float)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # Pre-processing
    parser.add_argument('--encoding_num', default="min_max_norm", type=str)
    parser.add_argument('--encoding_cat', default="onehot", type=str)  # onehot, hash or int for categorical attributes
    parser.add_argument('--num_hash_output', default=10, type=int)

    # pm4py
    parser.add_argument('--case_id_key', default="case_id", type=str)
    parser.add_argument('--activity_key', default="activity", type=str)

    # Parameters for gpu processing
    parser.add_argument('--gpu_ratio', default=0.2, type=float)
    parser.add_argument('--cpu_num', default=1, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
