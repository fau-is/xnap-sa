import numpy
from tensorflow.keras.models import load_model
from lime.lime_text import LimeTextExplainer
import xnap.nap.tester as test
from xnap.exp.lrp.util.heatmap import html_heatmap
import xnap.exp.lrp.util.browser as browser

# TODO choose consistent taxonomy -> case, trace or process_instance

# used for access in nested function "classifier_fn"
global args_
global preprocessor_
global event_log_

# used for string transformation
global context_enc_to_idx
global context_idx_to_enc
DELIMITER_WORD = " "
DELIMITER_ATTR = "_"
LEN_ACTIVITY_SYNTAX = 2
LEN_CONTEXT_SYNTAX = 3


def calc_and_plot_relevance_scores_instance(event_log, case, args, preprocessor):
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

    Returns
    -------

    """
    global args_
    global preprocessor_
    global event_log_
    args_ = args
    preprocessor_ = preprocessor
    event_log_ = event_log

    heatmap: str = ""
    for prefix_size in range(2, len(case)):

        # next activity prediction
        predicted_act_class, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist \
            = test.test_prefix(event_log, args, preprocessor, case, prefix_size)
        print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (
            prefix_size, predicted_act_class, target_act_class_str))
        print("Probability Distribution:")
        print(prob_dist)

        # LIME
        subseq_case = case[:prefix_size]
        subseq_case_str = transform_subseq_to_string(subseq_case, args, preprocessor)

        explainer = LimeTextExplainer()
        wrapped_classifier_function = wrapped_classifier_fn([subseq_case_str])  # function returns function!
        explanations = explainer.explain_instance(
                subseq_case_str,
                wrapped_classifier_function,
                num_features=len(subseq_case_str.split()),  # max number of features present in explanation, default=10
                num_samples=500                             # number of perturbed strings per prefix, default=5000
        )

        # heatmap
        prefix_words, R_words, R_words_context = create_heatmap_data(args, preprocessor, event_log, subseq_case,
                                                                     explanations, print_relevance_scores=True)
        heatmap = heatmap + html_heatmap(prefix_words, R_words, R_words_context) + "<br>"  # create heatmap
        if prefix_size == len(case)-1:
            browser.display_html(heatmap)  # display heatmap


def transform_subseq_to_string(subseq_case, args, preprocessor):
    """
    Transforms a subsequence of events (list) to a string (e.g. "0_1 0_0_2 0_1_1 1_1 1_0_1 1_1_3").
    Transformation scheme is explained in 'get_activity_str' and 'get_context_attribute_str'.

    Parameters
    ----------
    subseq_case : list of dicts, where single dict represents an event
        Subsequence / subset of a case whose length is prefix_size.
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    string : represents a subsequence of a case.

    """
    init_context_lookup(preprocessor)
    subseq_str_list = []
    for event_idx, event in enumerate(subseq_case):
        activity_str = str(get_activity_str(event, event_idx, args, preprocessor))
        context = []
        for context_name in preprocessor.context['attributes']:
            attr_str = get_context_attribute_str(event, event_idx, context_name, preprocessor)
            context.append(attr_str)
        context_str = DELIMITER_WORD.join(context)
        event_str = activity_str + DELIMITER_WORD + context_str
        subseq_str_list.append(event_str)
    subseq_str = DELIMITER_WORD.join(subseq_str_list)
    init_reverse_context_lookup()
    return subseq_str


def get_activity_str(event, event_idx, args, preprocessor):
    """
    Returns a string representing the activity of an event (e.g. "0_1").
    Transformation scheme:
    -> first digit (e.g. "0") indicates time step that this event represents (here it is the first event in a case)
    -> underscore ("_") separates time step indicator from identifier of activity
    -> second digit (e.g. "1") identifies the activity of this event (in respect to occurring activities in whole log)

    Parameters
    ----------
    event : dict
        A single event from a subsequence of a case.
    event_idx : int
        Time step at which this event occurs.
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    string : represents an activity on an event.

    """
    activity_encoding = tuple(event[args.activity_key])
    activity_id = preprocessor.activity['one_hot_to_event_ids'][activity_encoding]
    return str(event_idx) + DELIMITER_ATTR + str(activity_id)


def get_context_attribute_str(event, event_idx, attr_name, preprocessor):
    """
    Returns a string representing the value of a context attribute of an event (e.g. "1_0_2").
    Transformation scheme:
    -> first digit (e.g. "1") indicates time step that this event represents (here it is the second event in a case)
    -> first underscore ("_") separates time step indicator from identifier of context attribute
    -> second digit (e.g. "0") identifies the context attribute (name) (here it is the first context attr.)
    -> second underscore ("_") separates attribute indicator from identifier of attribute value
    -> third digit (e.g. "2") identifies the value for the referenced context attribute of this event (not in respect to
       occurrence in whole log, but numbering values as they appear in this exact subsequence; see context_enc_to_idx)

    Parameters
    ----------
    event : dict
        A single event from a subsequence of a case.
    event_idx : int
        Time step at which this event occurs.
    attr_name : str
        The name of an attribute.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    string : represents a context value on an event.

    """
    global context_enc_to_idx
    global context_idx_to_enc
    attr_idx = preprocessor.context['attributes'].index(attr_name)
    attr_enc = event[attr_name]
    if isinstance(attr_enc, list):
        attr_enc = tuple(attr_enc)
    if attr_enc not in context_enc_to_idx[attr_name].keys():
        attr_id = len(context_enc_to_idx[attr_name])  # new id
        # save mapping of initial encoding of context value to a (new) unique id
        context_enc_to_idx[attr_name][attr_enc] = attr_id
    else:
        # lookup assigned id for a encoded context value
        attr_id = context_enc_to_idx[attr_name][attr_enc]
    attr_str = str(event_idx) + DELIMITER_ATTR + str(attr_idx) + DELIMITER_ATTR + str(attr_id)
    return attr_str


def init_context_lookup(preprocessor):
    """
    Creates empty lookup dictionaries used to restore initial encoding of context values from string.

    Parameters
    ----------
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------

    """
    global context_idx_to_enc
    global context_enc_to_idx
    context_idx_to_enc = {}
    context_enc_to_idx = {}
    for context_name in preprocessor.context['attributes']:
        context_idx_to_enc[context_name] = {}
        context_enc_to_idx[context_name] = {}


def init_reverse_context_lookup():
    """
    Reverses filled lookup dictionary (to enable retrieval of initial context encoding from string).

    Returns
    -------

    """
    global context_idx_to_enc
    for context_name, mapping in context_enc_to_idx.items():
        for val, idx in mapping.items():
            context_idx_to_enc[context_name][idx] = val


def get_prefix_words(subseq_case, args, preprocessor):
    """
    Transforms a subsequence of encoded events into a list of events represented by their original value.

    Parameters
    ----------
    subseq_case : list of dicts, where single dict represents an event
        Subsequence / subset of a case whose length is prefix_size.
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.

    Returns
    -------
    list of lists, where a sublist list contains strings
        Sublists represent single events. Strings in a sublist represent original attribute values of this event.

    """
    subseq_str_list = []
    for event in subseq_case:
        event_list = []
        activity_enc = tuple(event[args.activity_key])
        activity_id = preprocessor.activity['one_hot_to_event_ids'][activity_enc]
        activity = preprocessor.unique_event_ids_map_to_name[activity_id]
        event_list.append(activity)
        for context_name in preprocessor.context['attributes']:
            context_enc = event[context_name]
            context_val = context_enc
            if isinstance(context_enc, list):
                context_val = preprocessor.context['enc_to_val'][context_name][tuple(context_enc)]
            event_list.append(context_val)
        subseq_str_list.append(event_list)
    return subseq_str_list


def create_heatmap_data(args, preprocessor, event_log, subseq_case, explanations, print_relevance_scores=False):
    """
    Prepares explanation data for heatmap visualization.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    preprocessor : nap.preprocessor.Preprocessor
        Object to preprocess input data.
    event_log : list of dicts, where single dict represents a case
        The initial event log.
    subseq_case : list of dicts, where single dict represents an event
        Subsequence / subset of a case whose length is prefix_size.
    explanations : lime.explanation.Explanation object
        Explanations object of lime returned by explain_instance. Contains relevance scores of attributes.

    Returns
    -------
    prefix_words : list of lists, where a sublist list contains strings
        Sublists represent single events. Strings in a sublist represent original attribute values of this event.
    R_words : ndarray with shape [1, max case length]
        Relevance scores of events in the subsequence to be explained.
    R_words_context : dict
        A entry in the dict contains relevance scores of context attributes (key is attribute name, value is array)

    """
    max_case_len = preprocessor.get_max_case_length(event_log)

    exp_dict = dict(explanations.as_list())
    R_act = {}
    R_context = {}
    if print_relevance_scores:
        relevance_scores = {}

    for str_key, rel_scr in exp_dict.items():
        len_attr_syntax = len(str_key.split(DELIMITER_ATTR))
        if len_attr_syntax == LEN_ACTIVITY_SYNTAX:
            # activity - look up information
            timestep = int(str_key.split(DELIMITER_ATTR)[0])
            R_act[timestep] = rel_scr

            if print_relevance_scores:
                activity_id = int(str_key.split(DELIMITER_ATTR)[1])
                activity = preprocessor.unique_event_ids_map_to_name[activity_id]
                if timestep not in relevance_scores:
                    relevance_scores[timestep] = {}
                relevance_scores[timestep][activity] = rel_scr
        else:
            # context - look up information
            timestep = int(str_key.split(DELIMITER_ATTR)[0])
            context_name = preprocessor.context['attributes'][int(str_key.split(DELIMITER_ATTR)[1])]
            if timestep not in R_context:
                R_context[timestep] = {}
            R_context[timestep][context_name] = rel_scr

            if print_relevance_scores:
                if timestep not in relevance_scores:
                    relevance_scores[timestep] = {}
                relevance_scores[timestep][context_name] = rel_scr

    # prefix_words
    prefix_words = get_prefix_words(subseq_case, args, preprocessor)

    # R_words
    R_words = numpy.zeros(max_case_len)
    for timestep, rel_scr in R_act.items():
        idx = (max_case_len - 1) - timestep
        R_words[idx] = rel_scr

    # R_words_context
    R_words_context = {}
    context_attributes = preprocessor.get_context_attributes()
    for attr in context_attributes:
        R_words_context[attr] = numpy.zeros(max_case_len)

    for timestep, t_dict in R_context.items():
        idx = (max_case_len - 1) - timestep
        for attr in t_dict.keys():
            R_words_context[attr][idx] = t_dict[attr]

    if print_relevance_scores:
        print("Relevance scores per timestep:")
        print(dict(sorted(relevance_scores.items(), key=lambda item: item[0]))) # sort by key (= time step)
        print("")

    return prefix_words, R_words, R_words_context


def wrapped_classifier_fn(subseq_str_list):
    """
    Performs preprocessing and prediction for use of LIME.

    Parameters
    ----------
    subseq_str_list : list of strings
        A list of strings (original or perturbed strings).

    Returns
    -------
    classifier_fn :
        Output of classifier_fn.

    """

    def classifier_fn(subseq_str_list):
        """
        Returns probability distribution of next activity prediction for a given subsequence of a case.

        Parameters
        ----------
        subseq_str_list : list of strings
            A list of strings (original or perturbed strings).

        Returns
        -------
        ndarray : shape [num_samples/perturbed strings, number of activities/classes in event log]
            Represents probability distributions of next activity predictions within lime (one probab. distr. per
            perturbed string/sequence of original string/subsequence).

        """
        model_index = 0
        model = load_model('%sca_%s_%s_%s.h5' % (
            args_.model_dir,
            args_.task,
            args_.data_set[0:len(args_.data_set) - 4], model_index))
        pred_probab = []
        for subseq_str in subseq_str_list:
            data_tensor = transform_string_to_tensor(subseq_str)
            y = model.predict(data_tensor)
            y = y[0][:]
            pred_probab.append(y)
        return numpy.asarray(pred_probab)

    def transform_string_to_tensor(subseq_str):
        """
        Produces a vector-oriented representation of feature data as a 3-dimensional tensor from a string.

        Parameters
        ----------
        subseq_str : str
            Represents a subsequence of a case.

        Returns
        -------
        ndarray : shape[1, T, F], 1 is number of samples, T is number of time steps, F is number of features.
            The features tensor.

        """
        num_features = preprocessor_.get_num_features()
        max_case_length = preprocessor_.get_max_case_length(event_log_)

        features_tensor = numpy.zeros((1,
                                       max_case_length,
                                       num_features), dtype=numpy.float32)

        subseq = get_event_lists_from_string(subseq_str)

        if len(subseq) > 0:
            # non-empty string (ignore perturbed strings that consist of only hidden features, i.e. '  ')
            subseq_enc = encode_subseq(subseq, preprocessor_)

            left_pad = max_case_length - len(subseq_enc)
            for timestep, event in enumerate(subseq_enc):
                # activity
                attr_enc = event[0]
                for idx, val in enumerate(attr_enc):
                    features_tensor[0, timestep + left_pad, idx] = val
                # context
                if preprocessor_.context_exists():
                    start_idx = 0
                    for context_enc in event[1:]:
                        if not isinstance(context_enc, list):
                            features_tensor[0, timestep + left_pad, start_idx +
                                            preprocessor_.get_length_of_activity_label()] = context_enc
                            start_idx += 1
                        else:
                            for idx, val in enumerate(context_enc, start=start_idx):
                                features_tensor[0, timestep + left_pad, idx +
                                                preprocessor_.get_length_of_activity_label()] = val
                            start_idx += len(context_enc)
        else:
            # if subseq is empty, return tensor which contains only zeros
            # TODO Sven - check if prediction of "zero tensor" (len(subseq) == 0) is supposed to be included in prediction in lines 392-394
            return features_tensor

        return features_tensor

    def get_event_lists_from_string(string):
        """
        Converts a string into a list of strings (words)

        Parameters
        ----------
        string : str
            Represents a subsequence of a case.

        Returns
        -------
        list of lists: a sublist list contains strings
            Sublists represent single events. Strings in a sublist represent attribute values of this event.

        """
        subseq = []
        str_list = [value for value in list(string.split(DELIMITER_WORD)) if value != ""]
        if len(str_list) > 0:
            # non-empty string (checks if perturbed strings consists of only hidden features, i.e. '  ')
            event = [str_list[0]]
            is_added = False
            prev_timestep = int(str_list[0].split(DELIMITER_ATTR)[0])
            for attr_val in str_list[1:]:
                is_added = False
                len_attr_syntax = len(attr_val.split(DELIMITER_ATTR))

                # activity
                if len_attr_syntax == LEN_ACTIVITY_SYNTAX:
                    # new event (indicated by time step)
                    subseq.append(event)
                    event = [attr_val]
                    prev_timestep = int(attr_val.split(DELIMITER_ATTR)[0])
                    is_added = True

                # context attribute
                if len_attr_syntax == LEN_CONTEXT_SYNTAX:
                    if int(attr_val.split(DELIMITER_ATTR)[0]) == prev_timestep:
                        event.append(attr_val)
                    else:
                        # if new event "starts" with context (activity of new event is hidden)
                        subseq.append(event)
                        event = [attr_val]
                        prev_timestep = int(attr_val.split(DELIMITER_ATTR)[0])

            if not is_added:
                subseq.append(event)
        return subseq

    def encode_subseq(subseq, preprocessor):
        """
        Restores encoding of all event attribute values.

        Parameters
        ----------
        subseq : list of lists, where a sublist list contains strings
            Sublists represent single events. Strings in a sublist represent attribute values of this event.
        preprocessor : nap.preprocessor.Preprocessor
            Object to preprocess input data.

        Returns
        -------
        list of lists, where a sublist list contains lists or floats
            Sublists represent single events. Lists in a sublist represent encoded categorical attribute values,
            floats represent numerical attribute values.

        """
        subseq_enc = []
        for event in subseq:
            event_enc = []
            len_attr_syntax = len(event[0].split(DELIMITER_ATTR))

            # activity
            activity_enc = get_dummy_activity()
            if len_attr_syntax == LEN_ACTIVITY_SYNTAX:
                # activity is present in perturbed event string, therefore overwrite activity dummy
                activity_id = int(event[0].split(DELIMITER_ATTR)[1])
                activity_enc = preprocessor.activity['event_ids_to_one_hot'][activity_id]
            event_enc.append(list(activity_enc))

            # context
            if preprocessor_.context_exists():
                context_enc = []
                for context_idx in range(len(preprocessor_.get_context_attributes())):
                    context_enc.append(get_dummy_context(context_idx))
                if len(event) > 1:
                    # if context attributes are present in perturbed string
                    context_in_str = [val for val in event[1:]]
                    for attr_val in context_in_str:
                        context_name_idx = int(attr_val.split(DELIMITER_ATTR)[1])
                        context_enc_idx = int(attr_val.split(DELIMITER_ATTR)[2])
                        context_name = preprocessor_.context['attributes'][context_name_idx]
                        context_encoding = context_idx_to_enc[context_name][context_enc_idx]
                        if isinstance(context_encoding, tuple):
                            context_encoding = list(context_encoding)
                        context_enc[context_name_idx] = context_encoding
                event_enc.extend(context_enc)
            subseq_enc.append(event_enc)

        return subseq_enc

    def get_dummy_activity():
        """
        Creates a dummy for a hidden activity in a perturbed string.

        Returns
        -------
        tuple of int : length of activity encoding (number of activities in log)
            Represents a hidden activity in a perturbed string (all values in this tuple representing a one hot encoded
            activity are zero).

        """
        return tuple([0] * preprocessor_.get_length_of_activity_label())

    def get_dummy_context(current_context_id):
        """
        Creates a dummy for a hidden context value in a perturbed string.

        Parameters
        ----------
        current_context_id : int
            Indicates a certain context attribute.

        Returns
        -------
        list of int : length of context encoding
            Represents a hidden context value in a perturbed string.

        """
        context_encoding_length = preprocessor_.context['encoding_lengths'][current_context_id]
        context_dummy = 0
        if context_encoding_length > 1:
            context_dummy = [0] * context_encoding_length
        return context_dummy

    return classifier_fn
