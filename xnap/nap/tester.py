from tensorflow.keras.models import load_model
import csv
import xnap.utils as utils
import xnap.nap.preprocessing.utilts as preprocessing_utils


def test_prefix(event_log, args, preprocessor, process_instance, prefix_size):
    """
    Perform test for LRP.

    :param event_log:
    :param args:
    :param preprocessor:
    :param process_instance:
    :param prefix_size:
    :return: parameters for LRP
    """

    # todo in case of cross validation select the model with the highest f1-score
    model_index = 0
    model = load_model('%sca_%s_%s_%s.h5' % (
                    args.model_dir,
                    args.task,
                    args.data_set[0:len(args.data_set) - 4], model_index))

    cropped_process_instance = process_instance[:prefix_size]
    cropped_process_instance_label = preprocessor.get_cropped_instance_label(prefix_size, process_instance)

    test_data = preprocessor.get_features_tensor(args, 'test', event_log, [cropped_process_instance])

    y = model.predict(test_data)
    y = y[0][:]

    prediction = preprocessor.get_predicted_label(y)

    prob_dist = dict()
    for index, prob in enumerate(y):
        prob_dist[preprocessor.get_event_type_from_event_id(index)] = y[index]

    test_data_reshaped = test_data.reshape(-1, test_data.shape[2])

    if cropped_process_instance_label == preprocessor.get_end_char():
        cropped_process_instance_label_id = preprocessor.get_event_id_from_event_name(cropped_process_instance_label)
    else:
        cropped_process_instance_label_id = preprocessor.get_event_id_from_one_hot(cropped_process_instance_label['event'])
        cropped_process_instance_label = preprocessor.get_event_type_from_event_id(cropped_process_instance_label_id)

    prefix_words = [preprocessor.get_event_type_from_event_id(preprocessor.get_event_id_from_one_hot(event['event'])) for event in cropped_process_instance]

    return preprocessor.get_event_type_from_event_id(preprocessor.get_event_id_from_one_hot(prediction)), \
            cropped_process_instance_label_id, \
            cropped_process_instance_label, \
            prefix_words, model, \
            test_data_reshaped, prob_dist


def test(args, preprocessor, event_log):
    """
    Perform test for model validation.
    :param event_log:
    :param args:
    :param preprocessor:
    :return: none
    """
    # TODO eliminate duplicated code fragment and export to preprocessor
    # get preprocessed data
    # similar to napt2.0tf evaluator l8
    train_indices, test_indices = preprocessing_utils.get_indices_split_validation(args, event_log)

    all_indices = []
    for case in event_log:
        all_indices.append(case.attributes['concept:name'])

    # similar to naptf2.0 trainer l11 ##TODO needs to be adopted towards split validation
    cases = preprocessor.get_cases_of_fold(event_log, [all_indices])  # TODO rename variable #ALL INDICES since we only got 1 split and want to use all indices in this one split

    # similar to nap2.0tf hpo l 62 ff
    test_cases = []
    for idx in test_indices:  # 0 because of no cross validation
        test_cases.append(cases[idx])

    model = load_model('%sca_%s_%s_%s.h5' % (
                    args.model_dir,
                    args.task,
                    args.data_set[0:len(args.data_set) - 4],
                    preprocessor.iteration_cross_validation))

    prediction_size = 1
    data_set_name = args.data_set.split('.csv')[0]
    # cases declared above
    result_dir_generic = './' + args.task + args.result_dir[1:] + data_set_name
    result_dir_fold = result_dir_generic + "_%d%s" % (
        preprocessor.iteration_cross_validation, ".csv")

    # start prediction
    with open(result_dir_fold, 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(
            ["CaseID", "Prefix length", "Ground truth", "Predicted"])

        # for prefix_size >= 1
        for prefix_size in range(1, preprocessor.get_max_case_length(event_log)):
            utils.llprint("Prefix size: %d\n" % prefix_size)

            for case in test_cases:
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
                        utils.llprint('-- End of case is predicted -- \n')
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