# Predicting Heart Disease from Health Survey Data (CDC BRFSS 2023)

Binary classification of self-reported heart attack or coronary heart disease from 37
survey answers, comparing traditional machine learning (scikit-learn) with deep learning
(TensorFlow/Keras with `tf.data`) on a large, imbalanced public health survey.

## Links

- Report: _to be added_
- Google Drive folder (zipped raw data and saved models): _to be added_
- Video presentation: _to be added_

## Headline results

Evaluated once on a held-out test set of 64,311 people (5,446 with heart disease, 8.5%).
Thresholds were chosen on a separate validation set and frozen.

| Model | ROC-AUC (95% CI) | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic regression | 0.847 (0.842-0.853) | 0.350 | 0.49 | 0.34 | 0.399 |
| Gradient boosting (HistGradientBoosting) | 0.851 (0.846-0.856) | 0.351 | 0.54 | 0.32 | 0.405 |
| Neural network (Keras Functional API) | 0.850 (0.845-0.855) | 0.353 | 0.54 | 0.33 | 0.406 |

Main findings:

- Three very different model families land within 0.004 ROC-AUC of each other, so the
  limit is the information in the survey answers and not the algorithm.
- Predicted probabilities are well calibrated, but even the highest-risk tenth of people
  is only about 36% positive.
- At one global threshold, recall is much lower for women (41% against 64% for men) and
  for people under 45 (7%), although ranking quality is similar across groups.
- The survey label is "ever told by a doctor", so it partly measures access to diagnosis.

## Data

CDC Behavioral Risk Factor Surveillance System, 2023 annual file
(<https://www.cdc.gov/brfss/annual_data/annual_2023.html>).

- 433,323 respondents and 350 columns in the raw file
- Target: yes to a heart attack (`CVDINFR4`) or coronary heart disease (`CVDCRHD4`)
- 428,738 rows have a determinable label; 8.47% are positive

The raw file is **not** stored in this repository. To reproduce:

1. Download `LLCP2023XPT.zip` from the link above and unzip it.
2. Place `LLCP2023.XPT` in the `data/` folder.

## Project layout

```
data/          raw file goes here (git-ignored)
notebooks/     01_heart_disease_brfss.ipynb, the full analysis
src/           data loading, feature cleaning, experiment runner, Keras models, evaluation
experiments/   results.csv (all 17 experiments), offline Weights & Biases logs, learning-curve data
reports/       figures used in the report
models/        saved best models (git-ignored; shared through the Drive folder)
```

## Reproducing

```
pip install -r requirements.txt
jupyter notebook notebooks/01_heart_disease_brfss.ipynb
```

Run the notebook top to bottom. The first cell block fixes all random seeds (`SEED = 42`).
Loading the raw file takes about 100 seconds the first time and is cached afterwards. A full
run, including all 17 experiments, takes roughly 20 minutes on a CPU.

Experiment logs are written to `experiments/wandb/` in offline mode (no account needed);
the same numbers appear in `experiments/results.csv`.

## Experiments

| ID | Approach | Purpose |
|---|---|---|
| E01 | Majority baseline | performance floor |
| E02-E04 | Logistic regression: default, class-weighted, without engineered features | linear baseline and ablations |
| E05-E06 | Decision tree: unconstrained, regularised | variance and regularisation |
| E07 | Random forest | variance reduction by averaging |
| E08-E10 | Gradient boosting: default, regularised, deeper | bias reduction and capacity |
| E11 | Boosting without engineered features | feature-engineering ablation |
| E12 | Boosting plus treatment variables | cost of the leakage exclusion |
| E13-E17 | Keras Sequential and Functional networks: depth, width, dropout, batch norm, L2, class weights, learning rate | deep-learning comparison |

## Notes on method

- Stratified 70/15/15 train/validation/test split with a fixed seed.
- All imputation, scaling and encoding is fitted on training rows only, inside the pipeline.
- Variables that are consequences of a heart event (medication, stroke history) and the
  CDC's own derived heart-disease flag are excluded, and the effect of excluding them is measured.
- Neural networks stop early on a slice of the training data, so validation data never
  influences training decisions.

## Limitations

Cross-sectional self-reported data from one US survey year; no causal interpretation;
the model was not validated on any population outside the BRFSS sample.
