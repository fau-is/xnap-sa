from xnap.exp.lrp.LSTM.LSTM_bidi import LSTM_bidi
import numpy as np
import tensorflow as tf


def calc_relevance_score_prefix(args, preprocessor, prefix_words, model, features_tensor_reshaped, tar_act_id):
    """
    Calculates relevance scores for all event attributes in a subsequence/prefix of a case.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    prefix_words : list of lists, where a sublist list contains strings
        Sublists represent single events. Strings in a sublist represent original attribute values of this event.
    model : tensorflow.python.keras.engine.functional.Functional object
        Deep neural network used for next activity prediction - to be explained.
    features_tensor_reshaped : ndarray with shape [max case length, num features]
        Encoded values for attributes of prefix.
    tar_act_id : int
        Id of target activity to be predicted.

    Returns
    -------
    R_words : ndarray with shape [1, max case length]
        Relevance scores of events in the subsequence to be explained.
    R_words_context : dict
        An entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)

    """
    # eager execution to avoid retracing with tensorflow
    tf.config.run_functions_eagerly(args.eager_execution)

    # perform LRP
    eps = 0.001  # small positive number, default = 0.001
    bias_factor = 0.0  # recommended value, default = 0.0
    net = LSTM_bidi(args, model, features_tensor_reshaped)  # load trained LSTM model
    Rx, Rx_rev, R_rest = net.lrp(prefix_words, tar_act_id, eps, bias_factor)

    # compute word-level LRP relevance
    num_activities = preprocessor.get_num_activities()
    R_words = np.sum(Rx[:, :num_activities] + Rx_rev[:, :num_activities], axis=1)
    R_words_context = {}

    current_col = num_activities
    for context_attribute in preprocessor.get_context_attributes():
        len_context_enc = preprocessor.get_length_of_context_encoding(context_attribute)
        R_words_context[context_attribute] = np.sum(Rx[:, current_col:current_col + len_context_enc] +
                                                    Rx_rev[:, current_col:current_col + len_context_enc], axis=1)
        current_col += preprocessor.get_length_of_context_encoding(context_attribute) + 1
    # scores = net.s.copy()  # classification prediction scores

    return R_words, R_words_context
