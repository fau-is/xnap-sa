# XNAP_SA: Explainable Next Activity Prediction for Service Analytics


## Setup of XNAP_SA
   1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) 
   2. After setting up miniconda you can make use of the `conda` command in your command line (e.g. CMD or Bash)
   3. To quickly install the `xnap_sa` package, run `pip install -e .` inside the root directory.
   4. To install required packages run `pip install -r requirements.txt` inside the root directory.
   6. Train and test the Bi-LSTM models for the next activity prediction by executing `runner.py` (note: you have to set in config.py the parameter "explain==False")
   7. Create explanations through LRP by executing `runner.py` (note: you have to set in config.py the parameter "explain==True")


## References

This repository belongs to the following paper:
```
@proceedings{weinzierl2024xnapsa,
    title={Context-aware Explanations of Accurate Predictions in Service Processes},
    author={Sven Weinzierl and Sandra Zilker and Jens Brunk and Kate Revoredo and Martin Matzner and Jörg Becker},
    booktitle={Proceedings of the 58th Hawaii International Conference on System Sciences},
    year={2024}
}
```
You can access the paper [here](https://www.researchgate.net/publication/374119620_Context-aware_Explanations_of_Accurate_Predictions_in_Service_Processes).

Further, this repository is an extension of the approach "explainable next activity prediction" (XNAP):
```
@proceedings{weinzierl2020xnap,
    title={XNAP: Making LSTM-based Next Activity Predictions Explainable by Using LRP},
    author={Sven Weinzierl and Sandra Zilker and Jens Brunk and Kate Revoredo and Martin Matzner and Jörg Becker},
    booktitle={Proceedings of the BPM 2020 International Workshops},
    year={2020}
}
```
You can access the paper [here](https://www.researchgate.net/publication/342918341_XNAP_Making_LSTM-based_Next_Activity_Predictions_Explainable_by_Using_LRP).
