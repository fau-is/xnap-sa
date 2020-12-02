# xnap-2.0: Explain LSTM-Models by Using LRP

## General
- Research goal:  Generate reliable explanations for next activity predictions

## Phases of the technique

1. Training and 

2. Testing Bi-LSTM Model

3. Generate Explanations
   
 
## Setting
- Task: nap
- Data attributes: yes
- Data sets: primary three bpi real-life data sets (both!); secondary synthetic data sets
- Metrics: auc_roc, acc, precision, recall, f1-score
- Model: Bi-LSTM
- Encoding: one-hot (activity and data attributes)
- Validation: 
    - split data set into train (80%) and test (20%); 
    - split train set (80%) into sub-train set (90%) and validation set (10%). 
- HPO: no (later yes)
- Shuffling: no
- Seed: 1377 (randomly selected, later no)
- Baseline: 
    - Bi-LSTM + LIME
    - Bi-LSTM + Shap
    - RF + LIME (optional)
    - RF + Shap (optional)


## Further details
- Ensuring reproducible results via a seed flag in config. Four seeds are set (holds only for cpu):
    - np.random.seed(1377)
    - tf.random.set_seed(1377)
    - random.seed(1377)
    - optuna.samplers.TPESampler(1377)


This repository comprise an extension of the approach "explainable next activity prediction" (xnap). If you make use of xnap, cite our previous paper:
```
@proceedings{weinzierl2020xnap,
    title={XNAP: Making LSTM-based Next Activity Predictions Explainable by Using LRP},
    author={Sven Weinzierl and Sandra Zilker and Jens Brunk and Kate Revoredo and Martin Matzner and Jörg Becker},
    booktitle={Proceedings of the 4th International Workshop on Artificial Intelligence for Business Process Management (AI4BPM2020)},
    year={2020}
}

```

You can access the paper [here](https://www.researchgate.net/publication/342918341_XNAP_Making_LSTM-based_Next_Activity_Predictions_Explainable_by_Using_LRP).
