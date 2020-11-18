import xnap.nap.tester as test
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
import numpy
from xnap.exp.lrp.util.heatmap import html_heatmap
import xnap.exp.lrp.util.browser as browser

# TODO add lime to requirements
# TODO choose consistent taxonomy -> case, trace or process_instance
# TODO check with log without context

global args_
global preprocessor_
global event_log_
context_enc_to_idx = {}
context_idx_to_enc = {}
DELIMITER_WORD = " "
DELIMITER_ATTR = "_"


def calc_and_plot_relevance_scores_instance(event_log, case, args, preprocessor):
    """ Calculates relevance scores and plots these scores in a heatmap. """
    # set for access through nested function "classifier_fn"
    global args_
    global preprocessor_
    global event_log_
    args_ = args
    preprocessor_ = preprocessor
    event_log_ = event_log

    heatmap: str = ""
    for prefix_size in range(2, len(case)):
        # next activity prediction
        predicted_act_class, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = test.test_prefix(
            event_log, args, preprocessor, case, prefix_size)
        print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (
            prefix_size, predicted_act_class, target_act_class_str))
        print("Probability Distribution:")
        print(prob_dist)

        # LIME
        subseq_case = case[:prefix_size]
        subseq_case_str = transform_subseq_to_string(subseq_case, args, preprocessor)

        explainer = LimeTextExplainer()  # TODO check if parameter class_names is necessary
        wrapped_classifier_function = wrapped_classifier_fn([subseq_case_str])  # function returns function!
        explanations = explainer.explain_instance(subseq_case_str,
                                                  wrapped_classifier_function,
                                                  num_samples=5  # number of perturbed strings per prefix, default=5000
                                                  )

        # heatmap
        prefix_words, R_words, R_words_context = create_heatmap_data(args, preprocessor, event_log, subseq_case,
                                                                     explanations)
        heatmap = heatmap + html_heatmap(prefix_words, R_words, R_words_context) + "<br>"  # create heatmap
        browser.display_html(heatmap)  # display heatmap

def transform_subseq_to_string(subseq_case, args, preprocessor):
    """
    Transforms a subsequence of events (list) to a string (e.g. "0_1 00_2 01_1 1_1 10_1 11_3").
    Transformation scheme is explained in 'get_activity_str' and 'get_context_attribute_str'.
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
    -> first digit (e.g. "0") indicates time step this event represents (in this case it is the first event in a case)
    -> underscore ("_") separates time step indicator from identifier of activity
    -> second digit (e.g. "1") identifies the activity of this event
    """
    activity_encoding = tuple(event[args.activity_key])
    activity_id = preprocessor.activity['one_hot_to_event_ids'][activity_encoding]
    return str(event_idx) + DELIMITER_ATTR + str(activity_id)

def get_context_attribute_str(event, event_idx, attr_name, preprocessor):
    """
    Returns a string representing the value of a context attribute of an event (e.g. "10_2").
    Transformation scheme:
    -> first digit (e.g. "1") indicates time step this event represents (in this case it is the second event in a case)
    -> second digit (e.g. "0") identifies the context attribute (name) (in this case it is the first context attr.)
    -> underscore ("_") separates time step indicator from identifier of context value
    -> third digit (e.g. "2") identifies the value for the referenced context attribute of this event
    """
    global context_enc_to_idx
    global context_idx_to_enc
    attr_idx = preprocessor.context['attributes'].index(attr_name)
    attr_enc = event[attr_name]
    attr_val = attr_enc
    if isinstance(attr_enc, list):
        attr_val = preprocessor.context['enc_to_val'][attr_name][tuple(attr_enc)]
    if attr_val not in context_enc_to_idx[attr_name].keys():
        attr_id = len(context_enc_to_idx[attr_name]) # new id
        # save mapping of initial encoding of context value to a (new) unique id
        context_enc_to_idx[attr_name][attr_val] = attr_id
    else:
        # lookup assigned id for a encoded context value
        attr_id = context_enc_to_idx[attr_name][attr_val]
    attr_str = str(event_idx) + str(attr_idx) + DELIMITER_ATTR + str(attr_id)
    return attr_str

def init_context_lookup(preprocessor):
    """ Creates empty lookup dictionaries used to restore initial encoding of context values from string. """
    global context_idx_to_enc
    global context_enc_to_idx
    context_idx_to_enc = {}
    context_enc_to_idx = {}
    for context_name in preprocessor.context['attributes']:
        context_idx_to_enc[context_name] = {}
        context_enc_to_idx[context_name] = {}

def init_reverse_context_lookup():
    """ Reverses filled lookup dictionary (to enable retrieval of initial context encoding from string). """
    global context_idx_to_enc
    for context_name, mapping in context_enc_to_idx.items():
        for val, id in mapping.items():
            context_idx_to_enc[context_name][id] = val

def get_prefix_words(subseq_case, args, preprocessor):
    """ Transforms a subsequence of encoded events into a list of events represented by their initial value. """
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

def create_heatmap_data(args, preprocessor, event_log, subseq_case, explanations):
    """ Prepares explanation data for heatmap visualization """
    max_case_len = preprocessor.get_max_case_length(event_log)

    exp_dict = dict(explanations.as_list())
    R_act = {}
    R_context = {}

    for str_key, rel_scr in exp_dict.items():
        attr_prefix = str_key.split(DELIMITER_ATTR)[0]
        if len(attr_prefix) == 1:
            # activity lookup information
            act_dict = {}
            act_dict['timestep'] = int(str_key.split(DELIMITER_ATTR)[0])
            act_dict['relevance'] = rel_scr
            R_act[str_key] = act_dict
        else:
            # context lookup information
            timestep = int(str_key.split(DELIMITER_ATTR)[0][0])
            context_dict = {}
            context_name = preprocessor.context['attributes'][int(str_key.split(DELIMITER_ATTR)[0][1])]
            context_dict['relevance'] = rel_scr
            if timestep not in R_context:
                R_context[timestep] = {}
            R_context[timestep][context_name] = context_dict

    # prefix_words
    prefix_words = get_prefix_words(subseq_case, args, preprocessor)

    # R_words TODO check if correct according to time steps
    R_words = numpy.zeros(max_case_len)
    for dict_timestep in R_act.values():
        idx = (max_case_len - 1) - dict_timestep['timestep']
        R_words[idx] = dict_timestep['relevance']

    # R_words_context # TODO it seems that there is a bug somewhere in generation of R_words_context
    R_words_context = {}
    context_attributes = preprocessor.get_context_attributes()
    for attr in context_attributes:
        R_words_context[attr] = numpy.zeros(max_case_len)

    for timestep, t_dict in R_context.items():
        idx = (max_case_len - 1) - timestep
        for attr in t_dict.keys():
            R_words_context[attr][idx] = t_dict[attr]['relevance']

    return prefix_words, R_words, R_words_context

def wrapped_classifier_fn(subseq_str_list):
    """ Performs preprocessing and prediction for use of LIME. """

    def classifier_fn(subseq_str_list):
        """ Returns probability distribution of next activity prediction for a given subsequence of a case. """
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
        """ Produces a vector-oriented representation of feature data as a 3-dimensional tensor from a string. """
        num_features = preprocessor_.get_num_features()
        max_case_length = preprocessor_.get_max_case_length(event_log_)

        features_tensor = numpy.zeros((1,
                                       max_case_length,
                                       num_features), dtype=numpy.float32)

        subseq = get_event_lists_from_string(subseq_str)
        if len(subseq) > 0: # TODO check this condition
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
                    for context_enc in event[1:]:
                        start_idx = 0
                        if not isinstance(context_enc, list):
                            features_tensor[0, timestep + left_pad, start_idx +
                                            preprocessor_.get_length_of_activity_label()] = context_enc
                            start_idx += 1
                        else:
                            for idx, val in enumerate(context_enc, start=start_idx):
                                features_tensor[0, timestep + left_pad, idx +
                                                preprocessor_.get_length_of_activity_label()] = val
                            start_idx += len(attr_enc)

        return features_tensor

    def get_event_lists_from_string(string):
        """ Converts a string into a list of strings (words)"""
        subseq = []
        str_list = [value for value in list(string.split(DELIMITER_WORD)) if value != ""]
        if len(str_list) > 0:
            # non-empty string (ignore perturbed strings that consist of only hidden features, i.e. '  ')
            event = [str_list[0]]
            is_added = False
            for attr_val in str_list[1:]:
                is_added = False
                attr_prefix = attr_val.split(DELIMITER_ATTR)[0]
                if len(attr_prefix) == 1:
                    subseq.append(event)
                    event = [attr_val]
                    is_added = True
                else:
                    event.append(attr_val)
            if not is_added:
                subseq.append(event)
        return subseq

    def encode_subseq(subseq, preprocessor):
        """ Restores encoding of all event attribute values. """
        subseq_enc = []
        for event in subseq:
            event_enc = []
            attribute_prefix = event[0].split(DELIMITER_ATTR)[0]

            # activity
            activity_enc = get_dummy_activity()
            if len(attribute_prefix) == 1:
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
                    for attr_idx, attr_val in enumerate(context_in_str):
                        context_name_idx = int(attr_val.split(DELIMITER_ATTR)[0][1])
                        context_name = preprocessor_.context['attributes'][context_name_idx]
                        context_enc_idx = int(attr_val.split(DELIMITER_ATTR)[1])
                        context_encoding = context_idx_to_enc[context_name][context_enc_idx]
                        context_enc[context_name_idx] = context_encoding
                event_enc.extend(context_enc)
                subseq_enc.append(event_enc)
        return subseq_enc

    def get_dummy_activity():
        """ Creates a dummy for a hidden activity in a perturbed string. """
        return tuple([0] * preprocessor_.get_length_of_activity_label())

    def get_dummy_context(current_context_id):
        """ Creates a dummy for a hidden context value in a perturbed string. """
        context_encoding_length = preprocessor_.context['encoding_lengths'][current_context_id]
        context_dummy = 0
        if context_encoding_length > 1:
            context_dummy = [0] * context_encoding_length
        return context_dummy

    return classifier_fn
