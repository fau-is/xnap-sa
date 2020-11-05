import xnap.nap.tester as test
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
import numpy
import re

# TODO add lime to requirements
# TODO choose consistent taxonomy -> case, trace or process_instance
# TODO is it possible to use LIME with context? (problem with multiple feature input because expected input format is string)

global args_
global preprocessor_
global event_log_

# string delimiters
DELIMITER_ATTR = "ATTRIBUTE"
DELIMITER_EVENT = "EVENT"
DELIMITER_ADD = " "

def calc_and_plot_relevance_scores_instance(event_log, case, args, preprocessor):

    # set for access through nested function "classifier_fn"
    global args_
    global preprocessor_
    global event_log_
    args_ = args
    preprocessor_ = preprocessor
    event_log_ = event_log

    for prefix_size in range(2, len(case)):
        # # next activity prediction
        # predicted_act_class, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = test.test_prefix(
        #     event_log, args, preprocessor, case, prefix_size)
        # print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (
        #     prefix_size, predicted_act_class, target_act_class_str))
        # print("Probability Distribution:")
        # print(prob_dist)

        # LIME
        subseq_case = case[:prefix_size]
        subseq_case_str = transform_subseq_to_string(subseq_case, args, preprocessor)
        class_names = list(preprocessor.activity['labels_to_chars'].keys())

        # TODO maybe use split_expression to correct perturbation (Docu "split_expression – strings will be split by this.")
        # https://github.com/marcotcr/lime/issues/27
        # bear in mind: dot in decimal numbers, DELIMITERs
        # split_expression = '|'.join(['\W+', 'word1', 'word2'])
        # split_expression = re.split(r'\W+', s)

        explainer = LimeTextExplainer(class_names=class_names) #, split_expression=split_expression)
        # wrapped_classifier_fn = get_predict_proba_fn_of_class([subseq_case_str]) # uncomment for better debugging of nested functions (if they are not nested/unindented)
        wrapped_classifier_function = wrapped_classifier_fn([subseq_case_str]) # function returns function!
        explanations = explainer.explain_instance(subseq_case_str, wrapped_classifier_function)

def transform_subseq_to_string(subseq_case, args, preprocessor):
    """ Transforms a subsequence of events (list) to a string. """
    subseq_str_list = []
    for event in subseq_case:
        activity_char = get_activity_char(event, args, preprocessor)
        context = []
        for context_name in preprocessor.context['attributes']:
            attr_str = get_context_attribute_str_value(event, context_name, preprocessor)

            # TODO remove (for testing purposes convert float to int as decimal 'dot' in float causes wrong perturbation)
            attr_idx = preprocessor.context['attributes'].index(context_name)
            attr_dtype = preprocessor.context['data_types'][attr_idx]
            if attr_dtype == 'num':
                # event_enc.append(float(attr_val))
                attr_str = str(int(float(attr_str)))

            context.append(attr_str)
        context_str = DELIMITER_ADD.join(context)
        # context_str = (DELIMITER_ADD + DELIMITER_ATTR + DELIMITER_ADD).join(context)
        event_str = activity_char + DELIMITER_ADD + context_str
        # event_str = activity_char + DELIMITER_ADD + DELIMITER_ATTR + DELIMITER_ADD + context_str
        subseq_str_list.append(event_str)
    subseq_str = " ".join(subseq_str_list)
    # subseq_str = (DELIMITER_ADD + DELIMITER_EVENT + DELIMITER_ADD).join(subseq_str_list)
    return subseq_str

def get_activity_char(event, args, preprocessor):
    """ Returns a char representing the activity of an event. """
    activity_encoding = tuple(event[args.activity_key])
    activity_char = preprocessor.activity['labels_to_chars'][activity_encoding]
    return activity_char

def get_context_attribute_str_value(event, attr_name, preprocessor):
    """ Returns original string value of a context attribute for an event. """
    attr_enc = event[attr_name]
    if isinstance(attr_enc, list):
        # categorical value -> retrieve original value
        attr_str = str(preprocessor.context['enc_to_val'][attr_name][tuple(attr_enc)])
    else:
        # numerical value -> keep encoded value
        attr_str = str(attr_enc)
    return attr_str

def wrapped_classifier_fn(subseq_str_list):
    """ Performs preprocessing and prediction for use of LIME. """

    def classifier_fn(subseq_str_list):
        """ Returns probability distribution as a result of next activity prediction for a given subsequence of a case. """
        model_index = 0
        model = load_model('%sca_%s_%s_%s.h5' % (
            args_.model_dir,
            args_.task,
            args_.data_set[0:len(args_.data_set) - 4], model_index))
        for subseq_str in subseq_str_list:
            # TODO create tensor from all perturbed strings?
            data_tensor = transform_string_to_tensor(subseq_str)
            y = model.predict(data_tensor)
            y = y[0][:]
            return y

    def transform_string_to_tensor(subseq_str):
        """ Produces a vector-oriented representation of feature data as a 3-dimensional tensor from a string. """
        num_features = preprocessor_.get_num_features()
        max_case_length = preprocessor_.get_max_case_length(event_log_)

        features_tensor = numpy.zeros((1,
                                       max_case_length,
                                       num_features), dtype=numpy.float32)

        subseq = get_event_lists_from_string(subseq_str)
        subseq_enc = encode_subseq(subseq, preprocessor_)

        left_pad = max_case_length - len(subseq_enc)
        for timestep, event in enumerate(subseq_enc):
            for attr_idx, attr_enc in enumerate(event):
                if attr_idx == 0:
                    # activity
                    for idx, val in enumerate(attr_enc):
                        features_tensor[0, timestep + left_pad, idx] = val
                else:
                    # context
                    start_idx = 0
                    if not isinstance(attr_enc, list):
                        features_tensor[
                            0, timestep + left_pad, start_idx + preprocessor_.get_length_of_activity_label()] = attr_enc
                        start_idx += 1
                    else:
                        for idx, val in enumerate(attr_enc, start=start_idx):
                            features_tensor[
                                0, timestep + left_pad, idx + preprocessor_.get_length_of_activity_label()] = val
                        start_idx += len(attr_enc)

        return features_tensor

    def get_event_lists_from_string(str):
        """ Converts a string into a list of strings (words)"""
        # str = str.replace(DELIMITER_ATTR, DELIMITER_ADD)
        str_list = [value for value in list(str.split(DELIMITER_ADD)) if value != DELIMITER_ATTR]
        subseq = []
        event = []
        for attr_idx, attr_val in enumerate(str_list):
            # if attr_val != DELIMITER_EVENT:
            if attr_idx == 0:
                # add first activity
                event.append(attr_val)
            else:
                if len(attr_val) == 1 and not attr_val.isnumeric():
                    # if attribute is activity (char)
                    subseq.append(event)
                    event = [attr_val]
                else:
                    # if attribute is context
                    event.append(attr_val)
        subseq.append(event)
        return subseq

    def encode_subseq(subseq, preprocessor):
        """ Restores encoding of all event attribute values. """
        subseq_enc = []
        event_enc = []
        for event in subseq:
            for attr_idx, attr_val in enumerate(event):
                context_idx = 0
                if attr_idx == 0:
                    activity_enc = preprocessor.activity['chars_to_labels'][attr_val]
                    event_enc.append(list(activity_enc))
                else:
                    # context
                    attr_dtype = preprocessor.context['data_types'][attr_idx - 1]
                    if attr_dtype == 'num':
                        # event_enc.append(float(attr_val))
                        event_enc.append(int(float(attr_val)))
                    else:
                        attr_name = preprocessor.context['attributes'][attr_idx - 1]
                        if attr_val.strip('-').isnumeric():
                            attr_val = int(attr_val)
                        attr_enc = preprocessor.context['val_to_enc'][attr_name][attr_val]
                        if isinstance(attr_enc, tuple):
                            attr_enc = list(attr_enc)
                        event_enc.append(attr_enc)
            subseq_enc.append(event_enc)
            event_enc = []
        return subseq_enc

    return classifier_fn