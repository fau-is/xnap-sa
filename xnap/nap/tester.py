import csv
import xnap.utils as utils
from datetime import datetime


def test(args, preprocessor, event_log, test_indices, output):
    """
    Perform test for model validation.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    test_indices : list of ints
        Indices of test cases in event log.
    output : dict
        Output for result evaluation.

    Returns
    -------
    """

    test_cases = preprocessor.get_subset_cases(args, event_log, test_indices)
    model = utils.load_nap_model(args)

    # start prediction
    prediction_distributions = []
    ground_truths = []

    with open(utils.get_output_path_predictions(args), 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(["CaseID", "Prefix length", "Ground truth activity", "Predicted activity"])

        prediction_size = 1
        # for prefix_size >= 1
        for prefix_size in range(1, preprocessor.get_max_case_length(event_log)):
            utils.ll_print("Prefix size: %d\n" % prefix_size)

            for case in test_cases:
                # 1.1.) prepare data: case subsequence
                subseq = get_case_subsequence(case, prefix_size)

                if contains_end_event(args, subseq, preprocessor):
                    # make no prediction for this subsequence since this case has ended already
                    continue

                ground_truth = get_ground_truth(args, case, prefix_size, prediction_size)
                ground_truths.append(ground_truth[0])
                prediction = []

                for current_prediction_size in range(prediction_size):
                    if current_prediction_size >= len(ground_truth):
                        continue

                    # 1.2.) prepare data: features tensor
                    features = preprocessor.get_features_tensor(args, event_log, [subseq])

                    if args.classifier == "RF" or args.classifier == "DT":
                        max_case_len = preprocessor.get_max_case_length(event_log)
                        num_features = preprocessor.get_num_features()
                        # flatten features tensor
                        features = features.reshape(len(features), max_case_len * num_features)

                    # 2.) make prediction
                    start_prediction_time = datetime.now()
                    predicted_label, predicted_dist = predict_label_and_dist(args, model, features, preprocessor)
                    output["prediction_times_seconds"] = (datetime.now() - start_prediction_time).total_seconds()
                    prediction.append(list(predicted_label))
                    prediction_distributions.append(predicted_dist)

                    if is_end_label(predicted_label, preprocessor):
                        utils.ll_print('-- End of case is predicted -- \n')
                        break

                # 3.) store prediction in file
                if len(ground_truth) > 0:
                    store_prediction(args, result_writer, case, prefix_size, ground_truth[0],
                                     prediction[0])

    return prediction_distributions, ground_truths


def test_prefix(event_log, args, preprocessor, case, prefix_size):
    """
    Perform test for LRP, LIME, or SHAP.

    :param event_log:
    :param args:
    :param preprocessor:
    :param case:
    :param prefix_size:
    :return: parameters for LRP, LIME, or SHAP
    """

    # ToDo in case of cross validation select the model with the highest f1-score
    subseq_case = case[:prefix_size]
    features_tensor = preprocessor.get_features_tensor(args, event_log, [subseq_case])

    target_act_label = tuple(case[prefix_size][args.activity_key])
    target_act_id = preprocessor.activity['labels_to_ids'][target_act_label]
    target_act_str = preprocessor.activity['ids_to_strings'][target_act_id]

    model = utils.load_nap_model(args)
    pred_prob = model.predict(features_tensor)[0]
    features_tensor_reshaped = features_tensor.reshape(-1, features_tensor.shape[2])

    pred_act_label = tuple(preprocessor.get_predicted_label(pred_prob))
    pred_act_id = preprocessor.activity['labels_to_ids'][pred_act_label]
    pred_act_str = preprocessor.activity['ids_to_strings'][pred_act_id]

    prob_dist = dict()
    for act_id, prob in enumerate(pred_prob):
        if act_id == len(pred_prob) - 1:
            act_name = preprocessor.get_end_char()
        else:
            act_name = preprocessor.activity['ids_to_strings'][act_id]
        prob_dist[act_name] = pred_prob[act_id]

    prefix_words = []
    for event in subseq_case:
        act_id = preprocessor.activity['labels_to_ids'][tuple(event[args.activity_key])]
        act_name = preprocessor.activity['ids_to_strings'][act_id]
        event_attr = [act_name]
        for context_attr_name in preprocessor.get_context_attributes():
            context_encoding = event[context_attr_name]
            if isinstance(context_encoding, list):
                # categorical attribute
                event_attr.append(preprocessor.context['one_hot_to_strings'][context_attr_name][tuple(context_encoding)])
            else:
                # numeric attribute
                event_attr.append(context_encoding)
        prefix_words.append(event_attr)

    return pred_act_str, target_act_id, target_act_str, prefix_words, model, features_tensor_reshaped, prob_dist


def test_manipulated_prefixes(args, preprocessor, event_log, manipulated_prefixes, test_indices):
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
    manipulated_prefixes : list of lists, where subists contain dicts (representing events of a subsequence of a case)
        Manipulated prefixes of test set.
    test_indices : list of ints
        Indices of test cases in event log.

    Returns
    -------

    """

    test_set = preprocessor.get_subset_cases(args, event_log, test_indices)
    model = utils.load_nap_model(args, preprocessor)

    # predict only next activity -> prediction_size = 1
    prediction_size = 1
    with open(utils.get_output_path_predictions(args, preprocessor), 'w') as result_file_fold:
        result_writer = csv.writer(result_file_fold, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(["CaseID", "Prefix length", "Ground truth", "Predicted"])

        results = {}
        for idx_case, manipulated_case in enumerate(manipulated_prefixes):
            for subseq in manipulated_case:
                prefix_size = len(subseq)
                case = test_set[idx_case]
                case_id = case[0].get(args.case_id_key)
                utils.ll_print("Prefix size: %d  Case ID: %d\n" % (prefix_size, case_id))

                if contains_end_event(args, subseq, preprocessor):
                    # make no prediction for this subsequence, since this case has ended already
                    continue

                ground_truth = get_ground_truth(args, case, prefix_size, prediction_size)

                prediction = []
                for current_prediction_size in range(prediction_size):
                    if current_prediction_size >= len(ground_truth):
                        continue

                    # 1. prepare data
                    features = preprocessor.get_features_tensor(args, event_log, [subseq])

                    # 2. make prediction
                    predicted_label = predict_label_and_dist(model, features, preprocessor)
                    prediction.append(list(predicted_label))

                    if is_end_label(predicted_label, preprocessor):
                        utils.ll_print('-- End of case is predicted -- \n')
                        break

                # 3. save prediction in certain format
                if len(ground_truth) > 0:
                    if prefix_size not in results:
                        results[prefix_size] = {}
                    results[prefix_size][case_id] = {}
                    results[prefix_size][case_id]['case'] = case
                    results[prefix_size][case_id]['ground_truth"'] = ground_truth[0]
                    results[prefix_size][case_id]['prediction'] = prediction[0]

        # 4. store prediction in file
        for prefix_size, case_dict in results.items():
            for case_id in case_dict.keys():
                store_prediction(args, result_writer, results[prefix_size][case_id]['case'], prefix_size,
                                 results[prefix_size][case_id]['ground_truth"'],
                                 results[prefix_size][case_id]['prediction'])


def get_case_subsequence(case, prefix_size):
    """ Crops a subsequence (= prefix) out of a whole case """
    return case[0:prefix_size]


def contains_end_event(args, subseq, preprocessor):
    """ Checks whether a subsequence of events contains an artificial end event, meaning case has ended """

    for event in subseq:

        act = event.get(args.activity_key)

        # is integer encoded
        if isinstance(act, int):
            act = [act]

        if is_end_label(tuple(act), preprocessor):

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

    ground_truth_events = case[prefix_size:prefix_size + prediction_size]
    ground_truth_activities = []
    for event in ground_truth_events:
        ground_truth_activities.append(event[args.activity_key])

    return ground_truth_activities


def predict_label_and_dist(args, model, features, preprocessor):
    """ Predicts and returns a label """

    if args.classifier == 'LSTM':
        Y = model.predict(features)
        predicted_dist = Y[0][:]
    else:
        # todo: predict_proba??
        Y = model.predict_proba(features)


    predicted_label = preprocessor.get_predicted_label(predicted_dist)

    return predicted_label, predicted_dist


def store_prediction(args, result_writer, case, prefix_size, ground_truth, prediction):
    """ Writes results into a result file """

    output = [case[0].get(args.case_id_key),
              prefix_size,
              str(ground_truth).encode("utf-8"),
              str(prediction).encode("utf-8")]

    result_writer.writerow(output)
