from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.models import load_model
import csv
import xnap.utils as utils


def test_prefix(args, preprocessor, process_instance, prefix_size):
    """
    Perform test for LRP.

    :param args:
    :param preprocessor:
    :param process_instance:
    :param prefix_size:
    :return: parameters for LRP
    """

    model_index = 0
    model = load_model('%sca_%s_%s_%s.h5' % (
                    args.model_dir,
                    args.task,
                    args.data_set[0:len(args.data_set) - 4], model_index))

    cropped_process_instance = preprocessor.get_cropped_instance(prefix_size, process_instance)
    cropped_process_instance_label = preprocessor.get_cropped_instance_label(prefix_size, process_instance)
    test_data = preprocessor.get_data_tensor_for_single_prediction(cropped_process_instance)

    y = model.predict(test_data)
    y = y[0][:]

    prediction = preprocessor.get_event_type_max_prob(y)

    prob_dist = dict()
    for index, prob in enumerate(y):
        prob_dist[preprocessor.get_event_type(index)] = y[index]

    test_data_reshaped = test_data.reshape(-1, test_data.shape[2])
    cropped_process_instance_label_id = preprocessor.data_structure['support']['map_event_type_to_event_id'][cropped_process_instance_label]

    return prediction, cropped_process_instance_label_id, cropped_process_instance_label, cropped_process_instance, model, test_data_reshaped, prob_dist


def test(args, preprocessor):
    """
    Perform test for model validation.
    :param args:
    :param preprocessor:
    :return: none
    """
    #TODO eliminate duplicated code fragment and export to preprocessor
    event_log = preprocessor.get_event_log(args)

    # get preprocessed data
    # similar to napt2.0tf evaluator l8
    train_index_per_fold, test_index_per_fold = preprocessor.get_indices_k_fold_validation(args, event_log)

    # similar to naptf2.0 trainer l11
    cases_of_fold = preprocessor.get_cases_of_fold(event_log, train_index_per_fold)

    model = load_model('%sca_%s_%s_%s.h5' % (
                    args.model_dir,
                    args.task,
                    args.data_set[0:len(args.data_set) - 4],
                    preprocessor.iteration_cross_validation))

    prediction_size = 1
    data_set_name = args.data_set.split('.csv')[0]
    #cases_of_fold declared above
    result_dir_generic = './' + args.task + args.result_dir[1:] + data_set_name
    result_dir_fold = result_dir_generic + "_%d%s" % (
        preprocessor.iteration_cross_validation, ".csv")

    # start prediction
    with open(result_dir_fold, 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ["CaseID", "Prefix length", "Ground truth", "Predicted"])

        # for prefix_size >= 2
        for prefix_size in range(2, preprocessor.get_max_case_length(event_log)):
            utils.llprint("Prefix size: %d\n" % prefix_size)


            for case in cases_of_fold:
                # 2.1. prepare data: case subsequence
                subseq = get_case_subsequence(case, prefix_size)

                if contains_end_event(args, subseq, preprocessor):
                    # make no prediction for this subsequence, since this case has ended already
                    continue

                ground_truth = get_ground_truth(args, case, prefix_size, prediction_size)
                prediction = []

                for current_prediction_size in range(prediction_size):
                    if current_prediction_size >= len(ground_truth):
                        continue

                    # 2.2. prepare data: features tensor
                    features = preprocessor.get_features_tensor(args, 'test', event_log, [subseq])

                    # 3. make prediction
                    predicted_label = predict_label(model, features, preprocessor)
                    prediction.append(list(predicted_label))

                    if is_end_label(predicted_label, preprocessor):
                        utils.llprint('! predicted, end of case ... \n')
                        break

                # 4. evaluate prediction
                if len(ground_truth) > 0:
                    document_and_evaluate_prediction(args, result_writer, case, prefix_size, ground_truth[0],
                                                     prediction[0])


            # for process_instance, event_id in zip(preprocessor.data_structure['data']['test']['process_instances'],
            #                                       preprocessor.data_structure['data']['test']['event_ids']):
            #
            #     cropped_process_instance = preprocessor.get_cropped_instance(
            #         prefix_size,
            #         process_instance)
            #
            #     if preprocessor.data_structure['support']['end_process_instance'] in cropped_process_instance:
            #         continue
            #
            #     ground_truth = ''.join(process_instance[prefix_size:prefix_size + prediction_size])
            #     prediction = ''
            #
            #     for i in range(prediction_size):
            #
            #         if len(ground_truth) <= i:
            #             continue
            #
            #         test_data = preprocessor.get_data_tensor_for_single_prediction(cropped_process_instance)
            #
            #         y = model.predict(test_data)
            #         y_char = y[0][:]
            #
            #         predicted_event = preprocessor.get_event_type_max_prob(y_char)
            #
            #         cropped_process_instance += predicted_event
            #         prediction += predicted_event
            #
            #         if predicted_event == preprocessor.data_structure['support']['end_process_instance']:
            #             print('! predicted, end of process instance ... \n')
            #             break
            #
            #     output = []
            #     if len(ground_truth) > 0:
            #         output.append(event_id)
            #         output.append(prefix_size)
            #         output.append(str(ground_truth).encode("utf-8"))
            #         output.append(str(prediction).encode("utf-8"))
            #         result_writer.writerow(output)

def get_case_subsequence(case, prefix_size):
    """ Crops a subsequence (= prefix) out of a whole case """
    return case._list[0:prefix_size]


def contains_end_event(args, subseq, preprocessor):
    """ Checks whether a subsequence of events contains an artificial end event, meaning case has ended """

    for event in subseq:
        if is_end_label(tuple(event.get(args.activity_key)), preprocessor):
            return True
        else:
            continue

    return False


def is_end_label(label, preprocessor):
    """ Checks whether event is an artificial end event """
    char = preprocessor.label_to_char(label)
    return char == preprocessor.get_end_char()


def get_ground_truth(args, case, prefix_size, prediction_size):
    """ Retrieves actual/true event label (= encoded activity) """

    ground_truth_events = case._list[prefix_size:prefix_size + prediction_size]
    ground_truth_activities = []
    for event in ground_truth_events:
        ground_truth_activities.append(event[args.activity_key])

    return ground_truth_activities


def predict_label(model, features, preprocessor):
    """ Predicts and returns a label """

    Y = model.predict(features)
    y_char = Y[0][:]
    predicted_label = preprocessor.get_predicted_label(y_char)

    return predicted_label


def document_and_evaluate_prediction(args, result_writer, case, prefix_size, ground_truth, prediction):
    """ Writes results into a result file """

    output = []
    output.append(case._list[0].get(args.case_id_key))
    output.append(prefix_size)
    output.append(str(ground_truth).encode("utf-8"))
    output.append(str(prediction).encode("utf-8"))

    result_writer.writerow(output)