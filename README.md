# xnap-2.0: Debugg and Improve LSTM-Models by Using LRP


## Steps of the technique
num_imp_steps <- 3
1. Initial step
    1. Train LSTM Model
    2. Produce Predictions

2. Improvement steps

for index in range(0, num_imp_steps):
    1. Produce Explainations via LRP
    2. Make interventions in the event log data (?)
    3. Re-train LSTM Model
    4. Produce Predictions

## Open questions?
- If we have the relevance, how do we intervene to the log? -> (e.g., change values, agument data etc.)
- Which information do we consider? -> (control-flow or context attributes) ->  

## Setting
- Metrics:
- Model: Bi-LSTM
- Encoding:
- Validation:
- HPO:
- Shuffling:
- Seed: 1377
- Baseline method for calculating relevance values: Bi-LSTM + LIME


This is the extension of xnap (Weinzierl et al. 2020).
```
@proceedings{weinzierl2020xnap,
    title={XNAP: Making LSTM-based Next Activity Predictions Explainable by Using LRP},
    author={Sven Weinzierl and Sandra Zilker and Jens Brunk and Kate Revoredo and Martin Matzner and Jörg Becker},
    booktitle={Proceedings of the 4th International Workshop on Artificial Intelligence for Business Process Management (AI4BPM2020)},
    year={2020}
}

```

You can access the paper [here](https://www.researchgate.net/publication/342918341_XNAP_Making_LSTM-based_Next_Activity_Predictions_Explainable_by_Using_LRP).
