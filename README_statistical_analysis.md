# Statistical Analysis for Hypoxia Model Evaluation

This repository contains comprehensive statistical analysis tools for evaluating and comparing machine learning models, with specific focus on hypoxia prediction models.

## Overview

The statistical analysis framework provides:

1. **McNemar's Test** - For comparing binary classifiers
2. **Statistical Significance Tests** - Chi-square, Mann-Whitney U, and more
3. **Model Performance Metrics** - Comprehensive evaluation metrics
4. **Visualization Tools** - Confusion matrices, ROC curves, PR curves
5. **Regression Analysis** - Statistical tests for regression models

## Files

### 1. `statistical_analysis.py`
Core statistical analysis class with methods for:
- McNemar's test implementation
- Confusion matrix analysis
- Statistical significance testing
- Model comparison visualization
- Regression statistical tests

### 2. `model_statistical_evaluation.py`
Integration script specifically for hypoxia models that:
- Loads trained models from saved files
- Generates predictions for comparison
- Runs comprehensive statistical evaluation
- Saves results and visualizations

### 3. `README_statistical_analysis.md`
This documentation file

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn torch
```

## Usage

### Basic Usage

```python
from statistical_analysis import StatisticalAnalyzer

# Create analyzer instance
analyzer = StatisticalAnalyzer()

# Perform McNemar's test
result = analyzer.mcnemar_test(y_true, y_pred1, y_pred2)
print(f"P-value: {result['p_value']:.6f}")
print(f"Significant: {result['is_significant']}")
```

### Complete Model Evaluation

```python
# Run complete evaluation on your hypoxia models
python model_statistical_evaluation.py
```

This will:
1. Load all available trained models
2. Generate predictions on test data
3. Perform statistical comparisons
4. Create visualizations
5. Save comprehensive results

## Statistical Tests Explained

### 1. McNemar's Test

**Purpose**: Tests whether two binary classifiers have significantly different error rates.

**When to use**: 
- Comparing two classification models
- Testing if one model is significantly better than another
- Analyzing model disagreement patterns

**Interpretation**:
- **p < 0.05**: Significant difference between models
- **p ≥ 0.05**: No significant difference
- Effect size indicates magnitude of difference

**Example Output**:
```
McNemar's Test p-value: 0.000123
Significant difference: True
Effect size: 0.0456
Interpretation: Significant difference with small effect size
```

### 2. Chi-Square Test

**Purpose**: Tests independence between two categorical variables.

**When to use**:
- Comparing model performance across different categories
- Testing if model errors are independent of data characteristics



### 4. Regression Statistical Tests

**Purpose**: Evaluate regression model performance and assumptions.

**Tests included**:
- Paired t-test for residuals
- Wilcoxon signed-rank test

## Output Files

The evaluation generates several output files in the `statistical_results/` directory:

### 1. `statistical_analysis_summary.txt`
Comprehensive text summary of all statistical tests and results.

### 2. `performance_comparison.csv`
Tabular comparison of model performance metrics.

### 3. `confusion_matrices.png`
Visualization of confusion matrices for all models.

### 4. `roc_comparison.png`
ROC curve comparison across all models.

### 5. `pr_comparison.png`
Precision-Recall curve comparison.

## Example Results Interpretation

### McNemar's Test Results

```
Model 1 vs Model 2:
  McNemar's Test p-value: 0.000123
  Significant difference: True
  Effect size: 0.0456
  Interpretation: Significant difference with small effect size
```

**What this means**:
- There is a statistically significant difference between the models
- The difference is small in magnitude
- Model 1 and Model 2 perform differently on the same data

### Performance Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | AUROC | AUPR |
|-------|----------|-----------|--------|----------|-------|------|
| LSTM  | 0.8234   | 0.7891    | 0.8123 | 0.8005   | 0.8567| 0.7891|
| STT   | 0.8156   | 0.7765    | 0.8034 | 0.7898   | 0.8432| 0.7654|

**What this shows**:
- LSTM performs slightly better than STT
- Both models have similar performance characteristics
- Statistical tests will determine if differences are significant

## Advanced Usage

### Custom Statistical Tests

```python
# Perform multiple test types
tests = ['mcnemar', 'chi2']
results = {}

for test_type in tests:
    result = analyzer.statistical_significance_test(
        y_true, y_pred1, y_pred2, test_type
    )
    results[test_type] = result
```

### Custom Visualization

```python
# Plot specific comparison types
analyzer.plot_model_comparison(
    y_true, 
    y_pred_dict, 
    'roc',  # or 'pr', 'residuals'
    save_path='custom_plot.png'
)
```

### Regression Model Analysis

```python
# For regression models
regression_results = analyzer.regression_statistical_tests(
    y_true, y_pred1, y_pred2
)

print(f"Paired t-test p-value: {regression_results['paired_t_test']['p_value']:.6f}")
print(f"Wilcoxon test p-value: {regression_results['wilcoxon_test']['p-value']:.6f}")
```

## Troubleshooting

### Common Issues

1. **No trained models found**
   - Ensure you've run `driver.py` first to train models
   - Check that model files exist in the current directory

2. **Data loading errors**
   - Verify `df_hyp_input.pkl` exists
   - Check data format and structure

3. **Import errors**
   - Ensure all required packages are installed
   - Check that `models.py` contains the required model classes

### Performance Tips

- Use GPU if available for faster model inference
- For large datasets, consider sampling for statistical tests
- Save intermediate results to avoid recomputation

## Statistical Significance Levels

- **α = 0.05**: Standard significance level (95% confidence)
- **α = 0.01**: High significance level (99% confidence)
- **α = 0.001**: Very high significance level (99.9% confidence)

## Effect Size Interpretation

- **< 0.1**: Small effect
- **0.1 - 0.3**: Medium effect
- **> 0.3**: Large effect

## Contributing

To extend the statistical analysis framework:

1. Add new test methods to `StatisticalAnalyzer` class
2. Implement corresponding visualization methods
3. Update the integration script if needed
4. Add comprehensive documentation

## References

- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages.
- Chi-square test for independence
- Mann-Whitney U test
- Statistical significance in machine learning

## License

This statistical analysis framework is provided as part of the hypoxia prediction project. 