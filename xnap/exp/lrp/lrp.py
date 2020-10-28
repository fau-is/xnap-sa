import xnap.nap.tester as test
from xnap.exp.lrp.LSTM.LSTM_bidi import LSTM_bidi
from xnap.exp.lrp.util.heatmap import html_heatmap, get_legend
import xnap.exp.lrp.util.browser as browser
import numpy as np


def calc_and_plot_relevance_scores_instance(event_log, trace, args, preprocessor):
    heatmap: str = ""

    #in head  "<style>" "</style>" could be placed in order to make div tags able to hover/unfold
    head_and_style = \
        "<!DOCTYPE html> <html lang=\"en\">" \
            "<head> " \
                "<meta charset=\"utf-8\"> " \
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0 \">" \
                "<title>XNAP2.0</title> " \
            "</head>" \
            "<body>"

    body_end = \
            "</body>" \
        "</html>"

    for idx in range(2, len(trace)):
        # next activity prediction
        #prefix words now is a 2d array of each event with its context attributes
        predicted_act_class, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = \
            test.test_prefix(event_log, args, preprocessor, trace, idx)
        print("Prefix: %s; Next activity prediction: %s; Next activity target: %s" % (
            idx, predicted_act_class, target_act_class_str))
        print("Probability Distribution:")
        print(prob_dist)

        # compute lrp relevances
        eps = 0.001  # small positive number
        bias_factor = 0.0  # recommended value
        net = LSTM_bidi(args, model, input_encoded)  # load trained LSTM model

        Rx, Rx_rev, R_rest = net.lrp(prefix_words, target_act_class, eps, bias_factor)  # perform LRP
        R_words = np.sum(Rx[:, :preprocessor.get_num_activities()] + Rx_rev[:, :preprocessor.get_num_activities()], axis=1)  # compute word-level LRP relevances for activity column
        R_words_context = {}
        column_names = [] #list of event and context attributes in order to print a legend
        column_names.append(args.activity_key)
        current_col = preprocessor.get_num_activities()
        for context_attribute in preprocessor.get_context_attributes():
            column_names.append(context_attribute)
            R_words_context[context_attribute] = np.sum(Rx[:, current_col:current_col + preprocessor.get_context_attribute_encoding_length(context_attribute)] + Rx_rev[:, current_col:current_col + preprocessor.get_context_attribute_encoding_length(context_attribute)], axis=1)
            current_col += preprocessor.get_context_attribute_encoding_length(context_attribute) + 1
        # scores = net.s.copy()  # classification prediction scores


        legend = "<br>" + "Legend: "
        legend += get_legend(column_names, R_words_context) + "<br>"

        heatmap += "<br>" + html_heatmap(prefix_words, R_words, R_words_context) + "<br>"  # create heatmap
        browser.display_html(head_and_style + legend + heatmap + body_end)  # display heatmap


def calc_relevance_scores_instance(trace, args, preprocessor):
    scores_prefix = list()

    for idx in range(2, len(trace)):
        # next activity prediction
        predicted_act_class, target_act_class, target_act_class_str, prefix_words, model, input_encoded, prob_dist = test.test_prefix(
            args, preprocessor, trace, idx)

        # compute lrp relevances
        eps = 0.001  # small positive number
        bias_factor = 0.0  # recommended value
        net = LSTM_bidi(args, model, input_encoded)  # load trained LSTM model

        Rx, Rx_rev, R_rest = net.lrp(prefix_words, target_act_class, eps, bias_factor)

        # todo: relevance scores for activity and attributes
        scores = np.sum(Rx + Rx_rev, axis=1)  # compute word-level LRP relevances

        scores_prefix.append(scores)

    return scores_prefix


def calc_relevance_scores_data_set():
    pass
