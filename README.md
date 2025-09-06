# Hypoxia Prediction Project

This project contains machine learning models for predicting hypoxia in ocean waters using various deep learning approaches.

## Main Python Files

### 1. `driver.py`
Main driver script for training and evaluating hypoxia prediction models.
- Trains LSTM, STT, Medformer, and TCN models
- Handles data preprocessing and feature engineering
- Generates model performance metrics and visualizations

**To run:**
```bash
python driver.py
```

### 2. `run_hypoxia_significance_tests.py`
Runs statistical significance tests on pre-trained hypoxia prediction models.

**To run:**
```bash
python run_hypoxia_significance_tests.py
```

### 3. `roms_hypoxia_comparison.py`
Creates ROMS (Regional Ocean Modeling System) comparison maps.

**To run:**
```bash
python roms_hypoxia_comparison.py
```


## Requirements

The project requires Python packages including:
- torch
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- cartopy (for geographic plots)
- xarray

## Output

Results are saved in various directories:
- `significance_test_results_*/` - Statistical test results
- `roms_hindcast_plots_*/` - ROMS comparison plots
- `saved_models/` - Trained model files
