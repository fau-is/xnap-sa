import xnap.nap.tester as test
from xnap.exp.lrp.LSTM.LSTM_bidi import LSTM_bidi
import numpy as np
import tensorflow as tf


def calc_relevance_score_prefix(args, preprocessor, event_log, case, prefix_size):
    """
    Calculates relevance scores for all event attributes in a subsequence/prefix of a case.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    case : dict
        A case from the event log.
    prefix_size : int
        Size of a prefix to be cropped out of a whole case.

    Returns
    -------
    prefix_words : list of lists, where a sublist list contains strings
        Sublists represent single events. Strings in a sublist represent original attribute values of this event.
    R_words : ndarray with shape [1, max case length]
        Relevance scores of events in the subsequence to be explained.
    R_words_context : dict
        An entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)
    column_names : list of str
        Names of attributes (activity + context) considered in prediction.

    """
    # eager execution to avoid retracing with tensorflow
    tf.config.run_functions_eagerly(args.eager_execution)

    # next activity prediction
    # prefix words now is a 2d array of each event with its context attributes
    predicted_act_class, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = \
        test.test_prefix(event_log, args, preprocessor, case, prefix_size)
    print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (
        prefix_size, predicted_act_class, target_act_class_str))
    print("Probability Distribution:")
    print(prob_dist)

    # compute relevance scores through lrp
    eps = 0.001  # small positive number
    bias_factor = 0.0  # recommended value
    net = LSTM_bidi(args, model, input_encoded)  # load trained LSTM model
    # perform LRP
    Rx, Rx_rev, R_rest = net.lrp(prefix_words, target_act_class, eps, bias_factor)

    num_activities = preprocessor.get_num_activities()

    # compute word-level LRP relevance for activity column
    R_words = np.sum(Rx[:, :num_activities] + Rx_rev[:, :num_activities], axis=1)
    R_words_context = {}

    column_names = [args.activity_key]  # list of event and context attributes in order to print a legend
    current_col = num_activities
    for context_attribute in preprocessor.get_context_attributes():
        column_names.append(context_attribute)
        len_context_enc = preprocessor.get_length_of_context_encoding(context_attribute)
        R_words_context[context_attribute] = np.sum(Rx[:, current_col:current_col + len_context_enc] +
                                                    Rx_rev[:, current_col:current_col + len_context_enc], axis=1)
        current_col += preprocessor.get_length_of_context_encoding(context_attribute) + 1
    # scores = net.s.copy()  # classification prediction scores

    return prefix_words, R_words, R_words_context, column_names
