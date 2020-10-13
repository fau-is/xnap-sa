# xnap-2.0: Explain and Improve LSTM-Models by Using LRP

## General
- Assumption: you can‘t get worse than before (iteration for iteration); change the underlying train data
- Research goal:  Learn better models with dedicated data augmentation (threshold?)

## Open questions/points?
- If we have the relevance, how do we intervene to the event log? -> (e.g., change values, agument data etc.)
- Which information do we consider? -> (control-flow or context attributes) -> 
- Specify in detail what the difference between model specific and model agnostic is (especially pros and cons; check literature)
- Potential problem: distorts predictions that were correct
- Should we show/measure the usability of explanations?

## Phases of the technique

num_imp_steps <- 3
1. Initial step
    1. Train LSTM Model
    
    2. Produce Predictions

2. Improvement steps
for index in range(0, num_imp_steps):

    1. Produce Explainations via LRP
    
    2. Make interventions (Algorithm learns form the manipulated data a model) 
        Assumption: we have an instance [A (0.9), B (0.1), C (0.1), D (0.9)] -> G, where E is the correct prediction and G the wrong
        
        2.1 Change instances in the event log data based on explanations
            (1) According to Rizzi et al. (2020): randomly change A and D -> violate process semantics?
            
        2.2 Augment instances in event log data based on explanations (we start with that)
            (1) Naive approach: 10 times add [A, B, C, D] to the event log
            (2) Advanced I: 10 times add [A, x, x, D] to the event log
            (3) Advanced II: 10 times add [x, B, C, x] to the event log
    
    3. Re-train LSTM Model
    
    4. Produce Predictions
    
 
## Setting
- Task: primary, nap; secondary outcome, time
- Data attributes: yes
- Data sets: primary three bpi real-life data sets (both!); secondary synthetic data sets
- Metrics:
- Model: Bi-LSTM
- Encoding:
- Validation: split data set into train (80%) and test (20%); split train set (80%) into sub-train set (90%) and validation set (10%). 
- HPO: no (at the beginning)
- Shuffling: no (yes)
- Seed: 1377 (randomly selected)
- Baseline: Bi-LSTM + LIME and Bi-LSTM + Shap


## Further details
- Ensuring reproducible results via a seed flag in config. Four seeds are set:
    - np.random.seed(1377)
    - tf.random.set_seed(1377)
    - random.seed(1377)
    - optuna.samplers.TPESampler(1377)


This repository comprise an extension of the approach "explainable next activity prediction" (xnap). If you make use of xnap, cite our paper:
```
@proceedings{weinzierl2020xnap,
    title={XNAP: Making LSTM-based Next Activity Predictions Explainable by Using LRP},
    author={Sven Weinzierl and Sandra Zilker and Jens Brunk and Kate Revoredo and Martin Matzner and Jörg Becker},
    booktitle={Proceedings of the 4th International Workshop on Artificial Intelligence for Business Process Management (AI4BPM2020)},
    year={2020}
}

```

You can access the paper [here](https://www.researchgate.net/publication/342918341_XNAP_Making_LSTM-based_Next_Activity_Predictions_Explainable_by_Using_LRP).
