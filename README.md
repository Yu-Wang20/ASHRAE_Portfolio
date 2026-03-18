# ASHRAE Great Energy Predictor III — Portfolio Project

Predict building energy consumption across 1,449 buildings using one year of
hourly meter readings (~20 million rows). This project demonstrates both
**rigorous statistical analysis** and **production-grade ML engineering**.

## Why This Project

The ASHRAE dataset is a compelling benchmark because it forces you to deal with
real-world messiness — missing weather data, meter-reading anomalies, building
heterogeneity, and temporal patterns at multiple scales. Tackling it end-to-end
exercises every stage of the data-science lifecycle: EDA, feature engineering,
model selection, and inference at scale.

## Hybrid Notebook + Modular Code Philosophy

This repository uses a two-layer design:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Notebooks** | `notebooks/` | Mathematical storytelling — derivations, visualisations, and narrative that explain *why* each decision was made. Designed to be read linearly. |
| **Modular scripts** | `src/` | Industrial-grade pipeline code that processes the full 20M+ row dataset. Importable, testable, and reproducible via config files. |

Notebooks import from `src/` so that every figure and table is backed by the
same code that runs at scale. This avoids the common pitfall of "works in
notebook, breaks in production."

## Setup

```bash
# Clone
git clone https://github.com/Yu-Wang20/ASHRAE_Portfolio.git
cd ASHRAE_Portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

Download the competition data from
[Kaggle](https://www.kaggle.com/competitions/ashrae-energy-prediction/data)
and place the CSV files in `data/`. See `data/README.md` for details.

## Project Structure

```
ASHRAE_Portfolio/
├── configs/
│   └── config.yaml          # Central configuration
├── data/
│   ├── README.md             # Data download instructions
│   └── *.csv                 # Raw competition data (git-ignored)
├── notebooks/                # Narrative & analysis notebooks
├── src/
│   └── __init__.py           # Modular pipeline code
├── outputs/
│   ├── models/               # Trained model artifacts
│   ├── figures/              # Saved plots
│   └── predictions/          # Submission files
├── requirements.txt
└── README.md
```

## License

This project is for educational and portfolio purposes.
