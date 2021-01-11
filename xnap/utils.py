import argparse
import sys
import csv
import sklearn
import arrow
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load


measures = {
    "accuracy_value": 0.0,
    "precision_micro_value": 0.0,
    "precision_macro_value": 0.0,
    "precision_weighted_value": 0.0,
    "recall_micro_value": 0.0,
    "recall_macro_value": 0.0,
    "recall_weighted_value": 0.0,
    "f1_micro_value": 0.0,
    "f1_macro_value": 0.0,
    "f1_weighted_value": 0.0,
    "auc_roc_value": 0.0,
    "training_time_seconds": 0.0,
    "prediction_times_seconds": 0.0,
    "explanation_times_seconds": 0.0
}


def load_output():
    return measures



def ll_print(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def str2bool(v):
    """
    Helper method.
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    """
    Cleans the measurement file.
    :param args:
    :return:
    """
    open(get_output_path_performance_measurements(args), "w").close()


def set_seed(args):
    """
    Sets seed for reproducible results.
    :param args: args.
    :return: none.
    """
    np.random.seed(args.seed_val)
    tf.random.set_seed(args.seed_val)


def calculate_measures(args, _measures):
    prefix = 0
    prefix_all_enabled = 1
    predicted_label = list()
    ground_truth_label = list()

    output_path = get_output_path_predictions(args)

    with open(output_path, 'r') as result_file:
        result_reader = csv.reader(result_file, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        next(result_reader)

        for row in result_reader:
            if not row:
                continue
            else:
                if int(row[1]) == prefix or prefix_all_enabled == 1:
                    ground_truth_label.append(row[2])
                    predicted_label.append(row[3])

    _measures["accuracy_value"] = sklearn.metrics.accuracy_score(ground_truth_label, predicted_label)
    _measures["precision_micro_value"] = sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='micro')
    _measures["precision_macro_value"] = sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='macro')
    _measures["precision_weighted_value"] = sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='weighted')
    _measures["recall_micro_value"] = sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='micro')
    _measures["recall_macro_value"] = sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='macro')
    _measures["recall_weighted_value"] = sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='weighted')
    _measures["f1_micro_value"] = sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='micro')
    _measures["f1_macro_value"] = sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='macro')
    _measures["f1_weighted_value"] = sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted')

    return _measures


def print_output(args, _output):

    ll_print("\nAccuracy: %f\n" % (_output["accuracy_value"]))

    ll_print("Precision (micro): %f\n" % (_output["precision_micro_value"]))
    ll_print("Precision (macro): %f\n" % (_output["precision_macro_value"]))
    ll_print("Precision (weighted): %f\n" % (_output["precision_weighted_value"]))

    ll_print("Recall (micro): %f\n" % (_output["recall_micro_value"]))
    ll_print("Recall (macro): %f\n" % (_output["recall_macro_value"]))
    ll_print("Recall (weighted): %f\n" % (_output["recall_weighted_value"]))

    ll_print("F1-Score (micro): %f\n" % (_output["f1_micro_value"]))
    ll_print("F1-Score (macro): %f\n" % (_output["f1_macro_value"]))
    ll_print("F1-Score (weighted): %f\n" % (_output["f1_weighted_value"]))

    if args.mode == 0:
        ll_print("Training time total: %f seconds\n" % (_output["training_time_seconds"]))
        ll_print("Prediction time avg: %f seconds\n" % (_output["prediction_times_seconds"]))
        ll_print("Prediction time total: %f seconds\n" % (_output["prediction_times_seconds"]))

    if args.mode == 2:
        ll_print("Explanation time avg: %f seconds\n" % (_output["explanation_times_seconds"]))
        ll_print("Explanation time total: %f seconds\n" % (_output["explanation_times_seconds"]))
    ll_print("\n")



def write_output(args, _measures):
    names = ["experiment",
             "mode",
             "validation",
             "accuracy",
             "precision",
             "recall",
             "f1-score"]

    if args.mode == 0:
        names.extend(["training-time-total", "prediction-time-avg", "prediction-time-total"])

    if args.mode == 2:
        names.extend(["explanation-time-avg", "explanation-time-total"])

    names.append("time-stamp")

    experiment = "%s-%s" % (args.data_set[:-4], args.dnn_architecture)
    mode = "split-%s" % args.split_rate_test

    values = [experiment, mode, "split-validation",
              _measures["accuracy_value"],
              _measures["precision_micro_value"],
              _measures["precision_macro_value"],
              _measures["precision_weighted_value"],
              _measures["recall_micro_value"],
              _measures["recall_macro_value"],
              _measures["recall_weighted_value"],
              _measures["f1_micro_value"],
              _measures["f1_macro_value"],
              _measures["f1_weighted_value"]]

    if args.mode == 0:
        values.append(_measures["training_time_seconds"])
        values.append(_measures["prediction_times_seconds"])
        values.append(_measures["prediction_times_seconds"])

    if args.mode == 2:
        values.append(_measures["explanation_times_seconds"])
        values.append(_measures["explanation_times_seconds"])
    values.append(arrow.now())

    output_path = get_output_path_performance_measurements(args)

    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
        if os.stat(output_path).st_size == 0:
            # If file is empty
            writer.writerow(names)
        writer.writerow(values)


def get_output_path_performance_measurements(args):

    directory = './%s%s' % (args.task, args.result_dir[1:])

    if args.mode == 0:
        file = 'measures_%s_%s.csv' % (args.data_set[:-4], args.classifier)
    if args.mode == 2:
        file = 'measures_%s_%s_manipulated.csv' % (args.data_set[:-4], args.classifier)

    return directory + file


def get_output_path_predictions(args):

    directory = './' + args.task + args.result_dir[1:]
    file = args.data_set.split(".csv")[0]

    file += "_0_%s" % args.classifier

    if args.mode == 2:
        file += "_manipulated"
    file += ".csv"

    return directory + file


def get_model_dir(args):
    """
    Returns the path to the stored trained model for the next activity prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.

    Returns
    -------
    str :
        Path to stored model.

    """
    model_dir = "%sca_%s_%s" % (args.model_dir, args.task, args.data_set[0:len(args.data_set) - 4])
    if args.classifier == "LSTM":
        model_dir += ".h5"
    else:
        model_dir += ".joblib"

    return model_dir


def load_nap_model(args):
    """
    Returns ML model used for next activity prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.

    Returns
    -------
    model : type depends on classifier type

    """

    model_dir = get_model_dir(args)
    if args.classifier == "LSTM":
        model = load_model(model_dir)
    else:
        model = load(model_dir)

    return model
