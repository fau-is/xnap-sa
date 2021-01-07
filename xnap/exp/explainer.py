import xnap.nap.tester as test
import xnap.exp.lrp.lrp as lrp
import xnap.exp.lime.lime as lime
import xnap.exp.shap.shap as shap
import xnap.exp.util.heatmap as heatmap_html
import xnap.exp.util.browser as browser
from datetime import datetime
import tensorflow as tf
import numpy as np


def calc_and_plot_relevance_scores_instance(event_log, case, args, preprocessor, train_indices):
    """
    Calculates relevance scores and plots these scores in a heatmap.

    Parameters
    ----------
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    case : dict
        A case from the event log.
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    train_indices : list of ints
        Indices of training cases from event log.

    Returns
    -------

    """
    heatmap = ""

    for prefix_size in range(2, len(case)):

        # next activity prediction
        pred_act_str, tar_act_id, tar_act_str, prefix_words, model, features_tensor_reshaped, prob_dist = \
            predict_next_activity_for_prefix(args, preprocessor, event_log, case, prefix_size)

        # explanations / computation of relevance of attributes
        if args.xai == "lrp":
            R_words, R_words_context = lrp.calc_relevance_score_prefix(args, preprocessor, prefix_words, model,
                                                                       features_tensor_reshaped, tar_act_id)

        if args.xai == "lime":
            R_words, R_words_context = lime.calc_relevance_score_prefix(args, preprocessor, event_log, case,
                                                                        prefix_size)

        if args.xai == "shap":
            background = shap.get_background_data(args, event_log, preprocessor, train_indices)
            R_words, R_words_context = shap.calc_relevance_score_prefix(args, preprocessor, event_log, case,
                                                                        prefix_size, background, model, pred_act_str,
                                                                        prefix_words)

        # heatmap
        heatmap = heatmap_html.add_relevance_to_heatmap(heatmap, prefix_words, R_words, R_words_context)
        if prefix_size == len(case) - 1:
            html = heatmap_html.create_html_heatmap_from_relevance_scores(args, preprocessor, heatmap, R_words_context)
            browser.display_html(html)


def get_manipulated_prefixes_from_relevance(args, preprocessor, event_log, train_indices, test_indices, output):
    """
    Returns manipulated prefixes - events are removed according to relevance scores.

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
    list of lists : where each list contains dicts (representing events of a subsequence of a case)
        Manipulated prefixes of test set.

    """
    test_set = preprocessor.get_subset_cases(args, event_log, test_indices)
    test_prefixes_manipulated = []

    if args.xai == "shap":
        background = shap.get_background_data(args, event_log, preprocessor, train_indices)

    for case in test_set:
        case_prefixes_manipulated = []

        for prefix_size in range(2, len(case)):

            # next activity prediction
            pred_act_str, tar_act_id, tar_act_str, prefix_words, model, features_tensor_reshaped, prob_dist = \
                predict_next_activity_for_prefix(args, preprocessor, event_log, case, prefix_size)

            # explanations / computation of relevance of attributes
            start_explanation_time = datetime.now()
            if args.xai == "lrp":
                R_words, R_words_context = lrp.calc_relevance_score_prefix(args, preprocessor, prefix_words, model,
                                                                           features_tensor_reshaped, tar_act_id)

            if args.xai == "lime":
                R_words, R_words_context = lime.calc_relevance_score_prefix(args, preprocessor, event_log, case,
                                                                            prefix_size)

            if args.xai == "shap":
                R_words, R_words_context = shap.calc_relevance_score_prefix(args, preprocessor, event_log, case,
                                                                            prefix_size, background, model,
                                                                            pred_act_str,
                                                                            prefix_words)

            avg_relevance_scores = get_avg_relevance_scores(R_words, R_words_context)
            case_prefixes_manipulated.append(get_manipulated_prefix(args, case[0:prefix_size], avg_relevance_scores))

            output["explanation_times_seconds"].append((datetime.now() - start_explanation_time).total_seconds())
        test_prefixes_manipulated.append(case_prefixes_manipulated)

    return test_prefixes_manipulated


def get_avg_relevance_scores(R_words, R_words_context):
    """
    Calculates the average of the relevance scores for activity and context attributes for each event in a subsequence.

    Parameters
    ----------
    R_words : ndarray with shape [1, max case length]
        Relevance scores of events in the subsequence to be explained.
    R_words_context : dict
        An entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)

    Returns
    -------
    ndarray: with shape [1, max case length]
        Average relevance scores for each event in the subsequence.

    """
    context_array = np.sum(np.array(list(R_words_context.values())), axis=0)
    sums = R_words + context_array
    avgs = sums / (1 + len(R_words_context))
    return avgs


def get_manipulated_prefix(args, prefix, avg_relevance_scores):
    """
    Removes events in a subsequence according to their average relevance scores - either those with highest or lowest
    average relevance scores (as specified in config).

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    prefix : list of dicts, where single dict represents an event
        Subsequence / subset of a case.
    avg_relevance_scores : ndarray with length max_case_len
        Average relevance scores for each event in the subsequence.

    Returns
    -------
    list of dicts : Subsequence of a case. Same as 'prefix' with n events removed according to their relevance scores.

    """
    manipulated_prefix = prefix[:]
    avg_idx_tuple = list(enumerate(avg_relevance_scores[-len(prefix):]))
    if args.removed_events_relevance == 'highest':
        avg_idx_tuple.sort(key=lambda t: t[1], reverse=True)
    elif args.removed_events_relevance == 'lowest':
        avg_idx_tuple.sort(key=lambda t: t[1])

    num_remove = args.removed_events_num
    while (len(prefix) - num_remove) <= 0:
        num_remove -= 1

    remove_indices = [idx for (idx, avg) in avg_idx_tuple[0:num_remove]]

    # remove in reverse order (not to interfere with subsequent indices)
    remove_indices.sort(reverse=True)
    for idx in remove_indices:
        del manipulated_prefix[idx]

    return manipulated_prefix


def predict_next_activity_for_prefix(args, preprocessor, event_log, case, prefix_size):

    if args.xai == "shap":
        # use TF 1.x as long as shap does not support TF >= 2.0
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()

    # next activity prediction
    pred_act_str, tar_act_id, tar_act_str, prefix_words, model, features_tensor_reshaped, prob_dist = \
        test.test_prefix(event_log, args, preprocessor, case, prefix_size)
    print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (prefix_size, pred_act_str,
                                                                                  tar_act_str))
    print("Probability Distribution:")
    print(prob_dist)

    return pred_act_str, tar_act_id, tar_act_str, prefix_words, model, features_tensor_reshaped, prob_dist
