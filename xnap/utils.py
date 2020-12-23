import argparse
import sys
import csv
import sklearn
import arrow
import os
import tensorflow as tf
import numpy as np

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
    open('./%s/results/output_%s.csv' % (args.task, args.data_set[:-4]), "w").close()


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

    result_dir_prefix = './' + args.task + args.result_dir[1:] + args.data_set.split(".csv")[0]
    if args.cross_validation:
        result_dir = result_dir_prefix + "_%d" % preprocessor.data_structure['support']['iteration_cross_validation']
    else:
        result_dir = result_dir_prefix + "_0"
    if args.mode == 2:
        result_dir += "_manipulated"
    result_dir += ".csv"

    with open(result_dir, 'r') as result_file:
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


def print_output(args, _output, index_fold):

    if args.cross_validation and index_fold < args.num_folds:
        llprint("\nAccuracy of fold %i: %f\n" % (index_fold, _output["accuracy_values"][index_fold]))
        llprint("Precision of fold %i: %f\n" % (index_fold, _output["precision_values"][index_fold]))
        llprint("Recall of fold %i: %f\n" % (index_fold, _output["recall_values"][index_fold]))
        llprint("F1-Score of fold %i: %f\n" % (index_fold, _output["f1_values"][index_fold]))
        if args.mode == 0:
            llprint("Training time of fold %i: %f seconds\n" % (index_fold, _output["training_time_seconds"][index_fold]))
            # TODO add prediction and explanation times if/when cross-validation is implemented

    else:
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
    If fold < max number of folds in cross validation than use a specific value, else avg works. In addition, this holds for split.
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

    if args.mode == 0:
        file_name = './%s%soutput_%s.csv' % (args.task, args.result_dir[1:], args.data_set[:-4])
    if args.mode == 2:
        file_name = './%s%soutput_%s_manipulated.csv' % (args.task, args.result_dir[1:], args.data_set[:-4])

    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_NONE, escapechar=' ')
        if os.stat(file_name).st_size == 0:
            # if file is empty
            writer.writerow(names)
        writer.writerow(values)
