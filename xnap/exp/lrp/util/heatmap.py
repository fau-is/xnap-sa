'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0+
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: see LICENSE file in repository root
'''

import matplotlib.pyplot as plt


def rescale_score_by_abs(score, max_score, min_score):
    """
    Normalize the relevance value (=score), accordingly to the extremal relevance values (max_score and min_score), 
    for visualization with a diverging colormap.
    i.e. rescale positive relevance to the range [0.5, 1.0], and negative relevance to the range [0.0, 0.5],
    using the highest absolute relevance for linear interpolation.
    """

    # CASE 1: positive AND negative scores occur --------------------
    if max_score > 0 and min_score < 0:

        if max_score >= abs(min_score):  # deepest color is positive
            if score >= 0:
                return 0.5 + 0.5 * (score / max_score)
            else:
                return 0.5 - 0.5 * (abs(score) / max_score)

        else:  # deepest color is negative
            if score >= 0:
                return 0.5 + 0.5 * (score / abs(min_score))
            else:
                return 0.5 - 0.5 * (score / min_score)

                # CASE 2: ONLY positive scores occur -----------------------------
    elif max_score > 0 and min_score >= 0:
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5 * (score / max_score)

    # CASE 3: ONLY negative scores occur -----------------------------
    elif max_score <= 0 and min_score < 0:
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5 * (score / min_score)


def getRGB(c_tuple):
    return "#%02x%02x%02x" % (int(c_tuple[0] * 255), int(c_tuple[1] * 255), int(c_tuple[2] * 255))


def div_event(w, score, colormap, scores_dict_context_attr, idx, max_s, min_s):

    context_attributes = ""
    for i, context_attr in enumerate(scores_dict_context_attr):
        score_context_attr = rescale_score_by_abs(scores_dict_context_attr[context_attr][len(scores_dict_context_attr[context_attr]) - idx - 1], max_s, min_s)
        context_attributes += get_div(w[i + 1], score_context_attr, colormap, get_context_attr_style())

    output_event_word = get_div(w[0] + context_attributes, score, colormap, get_event_attr_style())
    return output_event_word


def get_div(word, score, colormap, styles):
    if colormap == None:
        rgb = "#ffffff"
    else:
        rgb = getRGB(colormap(score))
    return "{}{}{}{}{}{}{}{}".format("<div ", " style=\"background-color:", rgb, ";", styles, "\">", word, "</div>")


def get_legend(column_names, scores_dict_context_attr):
    context_attributes = ""
    for i, context_attr in enumerate(scores_dict_context_attr):
        context_attributes += get_div(column_names[i + 1], 0, None, get_context_attr_style())
    output_event_word = get_div(column_names[0] + context_attributes, 0, None, get_event_attr_style())
    return output_event_word


def get_context_attr_style():
    return "margin: 5px; padding: 2px; border-style: solid; border-width: 1px; display:inline-block"

def get_event_attr_style():
    return "display:inline-block; border-style: solid; border-width: 1px;"

def html_heatmap(words, R_scores, R_scores_dict_context_attr, cmap_name="bwr"):
    """
    Return word-level heatmap in HTML format,
    with words being the list of words (as string),
    scores the corresponding list of word-level relevance values,
    and cmap_name the name of the matplotlib diverging colormap.
    Added Scores for context attribute in one hot encoding
    """

    colormap = plt.get_cmap(cmap_name)

    # assert len(words)==len(scores)
    max_context = 0
    min_context = 0
    for context_attr in R_scores_dict_context_attr:
        max_context = max(max_context, max(R_scores_dict_context_attr[context_attr]))
        min_context = min(min_context, min(R_scores_dict_context_attr[context_attr]))

    max_s = max(max(R_scores), max_context)
    min_s = min(min(R_scores), min_context)

    output_text = ""

    for idx, w in enumerate(words):
        score = rescale_score_by_abs(R_scores[len(R_scores) - idx - 1], max_s, min_s)
        output_text += div_event(w, score, colormap, R_scores_dict_context_attr, idx, max_s, min_s) + " "

    return output_text + "\n"
