# Data

## Download Instructions

1. Go to the ASHRAE Great Energy Predictor III competition page:
   https://www.kaggle.com/competitions/ashrae-energy-prediction/data
2. Accept the competition rules (if you haven't already).
3. Download the following files and place them in this directory:

   - `train.csv` — Hourly meter readings (target variable)
   - `building_metadata.csv` — Building characteristics
   - `weather_train.csv` — Weather data aligned with training period
   - `weather_test.csv` — Weather data aligned with test period
   - `test.csv` — Test set meter/building/timestamp combinations
   - `sample_submission.csv` — Submission format reference

Alternatively, use the Kaggle CLI:

```bash
kaggle competitions download -c ashrae-energy-prediction -p data/
unzip data/ashrae-energy-prediction.zip -d data/
```

## Note

These CSV files are git-ignored due to their size (~2 GB total).
Do not commit them to the repository.
