import xnap.nap.preprocessing.utilts as preprocessing_utils
import xnap.exp.lrp.lrp as lrp
import xnap.exp.lime.lime as lime
import numpy as np


def get_manipulated_test_prefixes_from_relevance(args, preprocessor, event_log):
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

    Returns
    -------
    list of lists : where each list contains dicts (representing events of a subsequence of a case)
        Manipulated prefixes of test set.

    """
    test_set = preprocessing_utils.get_test_set(args, event_log)
    test_prefixes_manipulated = []

    for case in test_set:
        case_prefixes_manipulated = []

        for prefix_size in range(2, len(case)):
            if args.xai == 'lrp':
                _, R_words, R_words_context, _ = lrp.calc_relevance_score_prefix(args, preprocessor, event_log, case,
                                                                                 prefix_size)
            if args.xai == 'lime':
                _, R_words, R_words_context, _ = lime.calc_relevance_score_prefix(args, preprocessor, event_log, case,
                                                                                  prefix_size)

            avg_relevance_scores = get_avg_relevance_scores(R_words, R_words_context)
            case_prefixes_manipulated.append(get_manipulated_prefix(args, case[0:prefix_size], avg_relevance_scores))

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
        A entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)

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
