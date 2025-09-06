#!/usr/bin/env python3
"""
Test script for statistical analysis functionality
This script demonstrates the statistical analysis tools with synthetic data
"""

import numpy as np
import pandas as pd
from statistical_analysis import StatisticalAnalyzer
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic data for testing statistical analysis
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (y_true, y_pred1, y_pred2, y_pred3)
    """
    np.random.seed(seed)
    
    # Generate true labels with some class imbalance
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Generate predictions from three different models
    # Model 1: Good performance
    y_pred1 = np.random.binomial(1, 0.8, n_samples)
    
    # Model 2: Slightly worse performance
    y_pred2 = np.random.binomial(1, 0.75, n_samples)
    
    # Model 3: Much worse performance
    y_pred3 = np.random.binomial(1, 0.6, n_samples)
    
    # Ensure some correlation with true labels
    y_pred1 = np.where(y_true == 1, 
                       np.random.binomial(1, 0.9, n_samples), 
                       np.random.binomial(1, 0.7, n_samples))
    
    y_pred2 = np.where(y_true == 1, 
                       np.random.binomial(1, 0.85, n_samples), 
                       np.random.binomial(1, 0.65, n_samples))
    
    y_pred3 = np.where(y_true == 1, 
                       np.random.binomial(1, 0.7, n_samples), 
                       np.random.binomial(1, 0.5, n_samples))
    
    return y_true, y_pred1, y_pred2, y_pred3

def test_basic_functionality():
    """Test basic statistical analysis functionality"""
    print("Testing Basic Statistical Analysis Functionality")
    print("=" * 60)
    
    # Generate synthetic data
    y_true, y_pred1, y_pred2, y_pred3 = generate_synthetic_data(1000)
    
    # Create analyzer instance
    analyzer = StatisticalAnalyzer()
    
    # Test 1: McNemar's test
    print("\n1. McNemar's Test (Model 1 vs Model 2):")
    mcnemar_result = analyzer.mcnemar_test(y_true, y_pred1, y_pred2)
    print(f"   Statistic: {mcnemar_result['statistic']:.4f}")
    print(f"   P-value: {mcnemar_result['p_value']:.6f}")
    print(f"   Significant: {mcnemar_result['is_significant']}")
    print(f"   Effect size: {mcnemar_result['effect_size']:.4f}")
    print(f"   Interpretation: {mcnemar_result['interpretation']}")
    
    # Test 2: Confusion matrix analysis
    print("\n2. Confusion Matrix Analysis:")
    cm_result1 = analyzer.confusion_matrix_analysis(y_true, y_pred1, "Model 1")
    cm_result2 = analyzer.confusion_matrix_analysis(y_true, y_pred2, "Model 2")
    cm_result3 = analyzer.confusion_matrix_analysis(y_true, y_pred3, "Model 3")
    
    print(f"   Model 1 Accuracy: {cm_result1['metrics']['accuracy']:.4f}")
    print(f"   Model 2 Accuracy: {cm_result2['metrics']['accuracy']:.4f}")
    print(f"   Model 3 Accuracy: {cm_result3['metrics']['accuracy']:.4f}")
    
    # Test 3: Statistical significance tests
    print("\n3. Statistical Significance Tests:")
    
    # Chi-square test
    chi2_result = analyzer.statistical_significance_test(y_true, y_pred1, y_pred2, 'chi2')
    print(f"   Chi-square p-value (Model 1 vs 2): {chi2_result['p_value']:.6f}")
    
    return analyzer, [cm_result1, cm_result2, cm_result3]

def test_visualizations(analyzer, cm_results):
    """Test visualization functionality"""
    print("\n4. Testing Visualizations:")
    print("   - Confusion matrices")
    print("   - ROC curves")
    print("   - Precision-Recall curves")
    
    # Generate probability-like scores for ROC/PR curves
    y_true, y_pred1, y_pred2, y_pred3 = generate_synthetic_data(1000)
    
    # Convert binary predictions to probabilities (for demonstration)
    y_prob1 = y_pred1.astype(float) + np.random.normal(0, 0.1, len(y_pred1))
    y_prob2 = y_pred2.astype(float) + np.random.normal(0, 0.1, len(y_pred2))
    y_prob3 = y_pred3.astype(float) + np.random.normal(0, 0.1, len(y_pred3))
    
    y_prob_dict = {
        'Model 1': y_prob1,
        'Model 2': y_prob2,
        'Model 3': y_prob3
    }
    
    # Plot confusion matrices
    try:
        analyzer.plot_confusion_matrices(cm_results, save_path='test_confusion_matrices.png')
        print("   ✓ Confusion matrices plotted successfully")
    except Exception as e:
        print(f"   ✗ Error plotting confusion matrices: {e}")
    
    # Plot ROC curves
    try:
        analyzer.plot_model_comparison(y_true, y_prob_dict, 'roc', save_path='test_roc_comparison.png')
        print("   ✓ ROC curves plotted successfully")
    except Exception as e:
        print(f"   ✗ Error plotting ROC curves: {e}")
    
    # Plot Precision-Recall curves
    try:
        analyzer.plot_model_comparison(y_true, y_prob_dict, 'pr', save_path='test_pr_comparison.png')
        print("   ✓ Precision-Recall curves plotted successfully")
    except Exception as e:
        print(f"   ✗ Error plotting PR curves: {e}")

def test_model_comparison_summary(analyzer, cm_results):
    """Test summary generation functionality"""
    print("\n5. Testing Summary Generation:")
    
    # Create sample results for summary
    sample_results = {
        'Model 1 Analysis': cm_results[0],
        'Model 2 Analysis': cm_results[1],
        'Model 3 Analysis': cm_results[2]
    }
    
    try:
        summary = analyzer.model_comparison_summary(sample_results)
        print("   ✓ Summary generated successfully")
        print("   Summary length:", len(summary), "characters")
        
        # Save summary to file
        with open('test_summary.txt', 'w') as f:
            f.write(summary)
        print("   ✓ Summary saved to 'test_summary.txt'")
        
    except Exception as e:
        print(f"   ✗ Error generating summary: {e}")

def test_regression_analysis():
    """Test regression statistical tests"""
    print("\n6. Testing Regression Analysis:")
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 500
    
    # True values
    y_true = np.random.normal(0, 1, n_samples)
    
    # Model predictions with different error patterns
    y_pred1 = y_true + np.random.normal(0, 0.3, n_samples)  # Good model
    y_pred2 = y_true + np.random.normal(0, 0.5, n_samples)  # Worse model
    
    analyzer = StatisticalAnalyzer()
    
    try:
        regression_results = analyzer.regression_statistical_tests(y_true, y_pred1, y_pred2)
        
        print("   ✓ Regression tests completed successfully")
        print(f"   Paired t-test p-value: {regression_results['paired_t_test']['p_value']:.6f}")
        print(f"   Wilcoxon test p-value: {regression_results['wilcoxon_test']['p_value']:.6f}")
        
    except Exception as e:
        print(f"   ✗ Error in regression analysis: {e}")

def main():
    """Main test function"""
    print("Statistical Analysis Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        analyzer, cm_results = test_basic_functionality()
        
        # Test visualizations
        test_visualizations(analyzer, cm_results)
        
        # Test summary generation
        test_model_comparison_summary(analyzer, cm_results)
        
        # Test regression analysis
        test_regression_analysis()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("Check the generated files for visualizations and results.")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 