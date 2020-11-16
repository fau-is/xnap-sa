import xnap.nap.tester as test
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
import numpy
from xnap.exp.lrp.util.heatmap import html_heatmap
import xnap.exp.lrp.util.browser as browser
import pandas as pd
import re

# TODO add lime to requirements
# TODO choose consistent taxonomy -> case, trace or process_instance
# TODO is it possible to use LIME with context? (problem with multiple feature input because expected input format is string)

global args_
global preprocessor_
global event_log_

DELIMITER = " "
prefix_lookup = {}

def calc_and_plot_relevance_scores_instance(event_log, case, args, preprocessor):

    # set for access through nested function "classifier_fn"
    global args_
    global preprocessor_
    global event_log_
    args_ = args
    preprocessor_ = preprocessor
    event_log_ = event_log

    heatmap: str = ""

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
        set_prefix_lookup(subseq_case)
        event_ids_str = DELIMITER.join([str(id) for id in prefix_lookup.keys()])

        explainer = LimeTextExplainer() # TODO check if class_names are necessary
        wrapped_classifier_function = wrapped_classifier_fn([event_ids_str])  # function returns function!
        explanations = explainer.explain_instance(event_ids_str,
                                                  wrapped_classifier_function,
                                                  num_samples=5  # number of perturbed strings per prefix, default=5000
                                                  )

        prefix_words, R_words, R_words_context = create_heatmap_data(args, preprocessor, event_log, subseq_case,
                                                                     explanations)

        # heatmap
        heatmap = heatmap + html_heatmap(prefix_words, R_words, R_words_context) + "<br>"  # create heatmap
        browser.display_html(heatmap)  # display heatmap


def transform_subseq_to_lists(subseq_case, args, preprocessor):
    """ Transforms a subsequence of events (list) to a list. """
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

def set_prefix_lookup(subseq_case):
    """ Maps events of a prefix to unique IDs. """
    global prefix_lookup
    for idx, event in enumerate(subseq_case):
        prefix_lookup[idx] = event

def create_heatmap_data(args, preprocessor, event_log, subseq_case, explanations):
    """ Prepares explanation data for heatmap visualization """
    exp_dict = dict(explanations.as_list())

    mapping_id_act = {}
    for event_id, event in prefix_lookup.items():
        mapping_id_act[event_id] = tuple(event[args_.activity_key])

    activities = []
    for event_id_str in exp_dict.keys():
        activities.append(mapping_id_act[int(event_id_str)])
    relevance_scores = list(exp_dict.values())
    relevance_dict = dict(zip(activities, relevance_scores))

    max_case_len = preprocessor.get_max_case_length(event_log)

    prefix_words = transform_subseq_to_lists(subseq_case, args, preprocessor)

    R_words = numpy.zeros(max_case_len)
    idx = max_case_len - 1
    for event in subseq_case:
        activity = event[args.activity_key]
        relevance_score = relevance_dict[tuple(activity)]
        R_words[idx] = relevance_score
        idx -= 1

    R_words_context = {}
    for context_attr in preprocessor.context['attributes']:
        R_words_context[context_attr] = numpy.zeros(max_case_len)

    return prefix_words, R_words, R_words_context

def wrapped_classifier_fn(subseq_str_list):
    """ Performs preprocessing and prediction for use of LIME. """

    def classifier_fn(subseq_str_list):
        """ Returns probability distribution as a result of next activity prediction for a given subsequence of a case. """
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

        left_pad = max_case_length - len(subseq_str)
        timestep = 0
        for event_id_str in subseq_str:
            if " " in event_id_str:
                continue
            event_id = int(event_id_str)
            event = prefix_lookup[event_id]

            # activity
            activity_enc = event[args_.activity_key]
            for idx, val in enumerate(activity_enc):
                features_tensor[0, timestep + left_pad, idx] = val

            # context
            if preprocessor_.context_exists():
                start_idx = 0
                for attribute_idx, attribute_key in enumerate(preprocessor_.context['attributes']):
                    attribute_values = event.get(attribute_key)

                    if not isinstance(attribute_values, list):
                        features_tensor[0, timestep + left_pad,
                                        start_idx + preprocessor_.get_length_of_activity_label()] = attribute_values
                        start_idx += 1
                    else:
                        for idx, val in enumerate(attribute_values, start=start_idx):
                            features_tensor[0, timestep + left_pad,
                                            idx + preprocessor_.get_length_of_activity_label()] = val
                        start_idx += len(attribute_values)

            timestep += 1

        return features_tensor

    return classifier_fn