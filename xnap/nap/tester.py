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

    test_data = preprocessor.get_features_tensor(args, event_log, [cropped_process_instance])

    y = model.predict(test_data)
    y = y[0][:]

    prediction = preprocessor.get_predicted_label(y)

    prob_dist = dict()
    for index, prob in enumerate(y):
        prob_dist[preprocessor.get_activity_type_from_activity_id(index)] = y[index]

    test_data_reshaped = test_data.reshape(-1, test_data.shape[2])

    if cropped_process_instance_label == preprocessor.get_end_char():
        #in case of max prefix length beeing reached c_p_i_l is the end char and does not need to be mapped
        cropped_process_instance_label_id = preprocessor.get_activity_id_from_activity_name(cropped_process_instance_label)
    else:
        cropped_process_instance_label_id = preprocessor.get_event_id_from_one_hot(cropped_process_instance_label[args.activity_key])
        cropped_process_instance_label = preprocessor.get_activity_type_from_activity_id(cropped_process_instance_label_id)

    prefix_words = []
    for event in cropped_process_instance:
        prefix_event_with_context = [
            preprocessor.get_activity_type_from_activity_id(preprocessor.get_event_id_from_one_hot(event[args.activity_key]))]
        for context_attr_name in preprocessor.get_context_attributes():
            # if attr is not categorial/one hot encoded then just returns the attribute value (numerical val)
            attr_type = str(type(event[context_attr_name]))
            if "float" in attr_type:
                prefix_event_with_context.append(event[context_attr_name])
            else:
                prefix_event_with_context.append(preprocessor.get_context_attribute_name_from_one_hot(context_attr_name, event[context_attr_name]))
        prefix_words.append(prefix_event_with_context)

    return preprocessor.get_activity_type_from_activity_id(preprocessor.get_event_id_from_one_hot(prediction)), \
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
    # todo eliminate duplicated code fragment and export to preprocessor
    if args.cross_validation:
        raise ValueError('cross_validation not yet implemented in XNAP2.0')
    else:
        test_cases = preprocessing_utils.get_test_set(args, event_log)

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
                    features = preprocessor.get_features_tensor(args, event_log, [subseq])

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


            # for process_instance, activity_id in zip(preprocessor.data_structure['data']['test']['process_instances'],
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
            #         output.append(activity_id)
            #         output.append(prefix_size)
            #         output.append(str(ground_truth).encode("utf-8"))
            #         output.append(str(prediction).encode("utf-8"))
            #         result_writer.writerow(output)


def test_manipulated_prefixes(args, preprocessor, event_log, manipulated_prefixes):
    """
    Performs next activity prediction on manipulated test set. Manipulated means that n events are removed from prefixes
    according to the respective relevance scores.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    manipulated_prefixes : TODO

    Returns
    -------

    """

    model = load_model(get_model_name(args, preprocessor))
    test_set = preprocessing_utils.get_test_set(args, event_log)

    # predict only next activity -> prediction_size = 1
    prediction_size = 1
    with open(get_result_dir_fold(args, preprocessor, manipulated=True), 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(["CaseID", "Prefix length", "Ground truth", "Predicted"])

        results = {}
        for idx_case, manipulated_case in enumerate(manipulated_prefixes):
            for subseq in manipulated_case:
                prefix_size = len(subseq)
                case = test_set[idx_case]
                case_id = case._list[0].get(args.case_id_key)
                utils.llprint("Prefix size: %d  Case ID: %d\n" % (prefix_size, case_id))

                if contains_end_event(args, subseq, preprocessor):
                    # make no prediction for this subsequence, since this case has ended already
                    continue

                ground_truth = get_ground_truth(args, case, prefix_size, prediction_size)
                prediction = []

                for current_prediction_size in range(prediction_size):
                    if current_prediction_size >= len(ground_truth):
                        continue

                    # 2.2. prepare data: features tensor
                    features = preprocessor.get_features_tensor(args, event_log, [subseq])

                    # 3. make prediction
                    predicted_label = predict_label(model, features, preprocessor)
                    prediction.append(list(predicted_label))

                    if is_end_label(predicted_label, preprocessor):
                        utils.llprint('-- End of case is predicted -- \n')
                        break

                # 4. store prediction
                if len(ground_truth) > 0:
                    if prefix_size not in results:
                        results[prefix_size] = {}
                    results[prefix_size][case_id] = {}
                    results[prefix_size][case_id]['case'] = case
                    results[prefix_size][case_id]['ground_truth"'] = ground_truth[0]
                    results[prefix_size][case_id]['prediction'] = prediction[0]

        for prefix_size, case_dict in results.items():
            for case_id in case_dict.keys():
                document_and_evaluate_prediction(args,
                                                 result_writer,
                                                 results[prefix_size][case_id]['case'],
                                                 prefix_size,
                                                 results[prefix_size][case_id]['ground_truth"'],
                                                 results[prefix_size][case_id]['prediction'])


def get_model_name(args, preprocessor):
    """
    Returns the name of the stored trained model for the next activity prediction.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    str :
        Name of the model.

    """
    return '%sca_%s_%s_%s.h5' % (args.model_dir,
                                 args.task,
                                 args.data_set[0:len(args.data_set) - 4],
                                 preprocessor.iteration_cross_validation)


def get_result_dir_fold(args, preprocessor, manipulated=False):
    """
    Returns result directory of a fold.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    Returns
    -------
    str :
        Directory of the result file for the current fold.
    """

    data_set_name = args.data_set.split('.csv')[0]
    result_dir_generic = './' + args.task + args.result_dir[1:] + data_set_name
    result_dir_fold = result_dir_generic + "_%d%s" % (preprocessor.iteration_cross_validation, ".csv")
    if manipulated:
        result_dir_fold = result_dir_generic + "_%d_%s%s" % (preprocessor.iteration_cross_validation, "manipulated", ".csv")

    return result_dir_fold


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