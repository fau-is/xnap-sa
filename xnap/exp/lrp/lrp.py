import xnap.nap.tester as test
from xnap.exp.lrp.LSTM.LSTM_bidi import LSTM_bidi
from xnap.exp.lrp.util.heatmap import html_heatmap
import xnap.exp.lrp.util.browser as browser
import numpy as np


def calc_and_plot_relevance_scores_instance(event_log, trace, args, preprocessor):
    heatmap: str = ""

    for idx in range(2, len(trace)):
        # next activity prediction
        # TODO check for context attribute since lrp is a bit more complex
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
        R_words = np.sum(Rx[:, :preprocessor.get_num_activities()] + Rx_rev[:, :preprocessor.get_num_activities()], axis=1)  # compute word-level LRP relevances
        # scores = net.s.copy()  # classification prediction scores

        """
        prefix_words_list = []
        for prefix in prefix_words:
            prefix_words_list.append(preprocessor.get_event_type_from_event_id(preprocessor.get_event_id_from_one_hot(prefix['event'])))
        """

        heatmap = heatmap + html_heatmap(prefix_words, R_words) + "<br>"  # create heatmap
        browser.display_html(heatmap)  # display heatmap


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
