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


output = {
    "accuracy_values": [],
    "precision_values": [],
    "recall_values": [],
    "f1_values": [],
    "training_time_seconds": [],
    "prediction_times_seconds": [],
    "explanation_times_seconds": []
}


def load_output():
    return output


def avg(numbers):
    if len(numbers) == 0:
        return sum(numbers)

    return sum(numbers) / len(numbers)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clear_measurement_file(args):
    open(get_output_path_performance_measurements(args), "w").close()


def set_seed(args):
    """
    Sets seed for reproducible results.
    :param args: args.
    :return: none.
    """
    np.random.seed(args.seed_val)
    tf.random.set_seed(args.seed_val)


def get_output(args, preprocessor, _output):
    prefix = 0
    prefix_all_enabled = 1

    predicted_label = list()
    ground_truth_label = list()

    output_path = get_output_path_predictions(args, preprocessor)

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

    _output["accuracy_values"].append(sklearn.metrics.accuracy_score(ground_truth_label, predicted_label))
    _output["precision_values"].append(
            sklearn.metrics.precision_score(ground_truth_label, predicted_label, average='weighted'))
    _output["recall_values"].append(
            sklearn.metrics.recall_score(ground_truth_label, predicted_label, average='weighted'))
    _output["f1_values"].append(sklearn.metrics.f1_score(ground_truth_label, predicted_label, average='weighted'))

    return _output


def print_output(args, _output):

    llprint("\nAccuracy avg: %f\n" % (avg(_output["accuracy_values"])))
    llprint("Precision avg: %f\n" % (avg(_output["precision_values"])))
    llprint("Recall avg: %f\n" % (avg(_output["recall_values"])))
    llprint("F1-Score avg: %f\n" % (avg(_output["f1_values"])))

    if args.mode == 0:
        llprint("Training time total: %f seconds\n" % (avg(_output["training_time_seconds"])))
        llprint("Prediction time avg: %f seconds\n" % (avg(_output["prediction_times_seconds"])))
        llprint("Prediction time total: %f seconds\n" % (sum(_output["prediction_times_seconds"])))

    if args.mode == 2:
        llprint("Explanation time avg: %f seconds\n" % (avg(_output["explanation_times_seconds"])))
        llprint("Explanation time total: %f seconds\n" % (sum(_output["explanation_times_seconds"])))

    llprint("\n")


def get_mode(index_fold, args):
    if index_fold == -1:
        return "split-%s" % args.split_rate_test
    elif index_fold != args.num_folds:
        return "fold%s" % index_fold
    else:
        return "avg"


def get_output_value(_mode, _index_fold, _output, measure, args):
    """
    If fold < max number of folds in cross validation than use a specific value, else avg works. In addition, this holds
    for split.
    """

    if _mode != "split-%s" % args.split_rate_test and _mode != "avg" and _mode != "sum":
        return _output[measure][_index_fold]
    else:
        if _mode == "sum":
            return sum(_output[measure])
        else:
            return avg(_output[measure])


def write_output(args, _output, index_fold):
    names = ["experiment", "mode", "validation", "accuracy", "precision", "recall", "f1-score"]
    if args.mode == 0:
        names.extend(["training-time-total", "prediction-time-avg", "prediction-time-total"])
    if args.mode == 2:
        names.extend(["explanation-time-avg", "explanation-time-total"])
    names.append("time-stamp")

    experiment = "%s-%s" % (args.data_set[:-4], args.dnn_architecture)
    mode = get_mode(index_fold, args)
    values = [experiment, mode]
    if args.cross_validation:
        values.append("cross-validation")
    else:
        values.append("split-validation")
    values.append(get_output_value(mode, index_fold, _output, "accuracy_values", args))
    values.append(get_output_value(mode, index_fold, _output, "precision_values", args))
    values.append(get_output_value(mode, index_fold, _output, "recall_values", args))
    values.append(get_output_value(mode, index_fold, _output, "f1_values", args))
    if args.mode == 0:
        values.append(get_output_value(mode, index_fold, _output, "training_time_seconds", args))
        values.append(get_output_value(mode, index_fold, _output, "prediction_times_seconds", args))
        values.append(get_output_value("sum", index_fold, _output, "prediction_times_seconds", args))
    if args.mode == 2:
        values.append(get_output_value(mode, index_fold, _output, "explanation_times_seconds", args))
        values.append(get_output_value("sum", index_fold, _output, "explanation_times_seconds", args))
    values.append(arrow.now())

    output_path = get_output_path_performance_measurements(args)

    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
        if os.stat(output_path).st_size == 0:
            # if file is empty
            writer.writerow(names)
        writer.writerow(values)


def get_output_path_performance_measurements(args):

    directory = './%s%s' % (args.task, args.result_dir[1:])

    if args.mode == 0:
        file = 'output_%s_%s.csv' % (args.data_set[:-4], args.classifier)
    if args.mode == 2:
        file = 'output_%s_%s_manipulated.csv' % (args.data_set[:-4], args.classifier)

    return directory + file


def get_output_path_predictions(args, preprocessor):

    directory = './' + args.task + args.result_dir[1:]
    file = args.data_set.split(".csv")[0]

    file += "_0_%s" % args.classifier

    if args.mode == 2:
        file += "_manipulated"
    file += ".csv"

    return directory + file


def get_model_dir(args, preprocessor):
    """
    Returns the path to the stored trained model for the next activity prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    str :
        Path to stored model.

    """
    model_dir = "%sca_%s_%s_%s" % (args.model_dir, args.task, args.data_set[0:len(args.data_set) - 4],
                                   preprocessor.iteration_cross_validation)
    if args.classifier == "DNN":
        model_dir += ".h5"
    else:
        model_dir += ".joblib"

    return model_dir


def load_nap_model(args, preprocessor):
    """
    Returns ML model used for next activity prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    model : type depends on classifier type

    """

    model_dir = get_model_dir(args, preprocessor)
    if args.classifier == "DNN":
        model = load_model(model_dir)
    else:
        model = load(model_dir)

    return model
