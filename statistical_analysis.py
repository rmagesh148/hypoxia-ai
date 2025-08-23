import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2, chi2_contingency, wilcoxon, ttest_ind, ttest_rel
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for machine learning model evaluation
    Includes McNemar's test, statistical significance tests, and model comparison methods
    """
    
    def __init__(self):
        self.results = {}
        
    def mcnemar_test(self, y_true, y_pred1, y_pred2, alpha=0.05, correction=True):
        """
        Perform McNemar's test to compare two binary classifiers
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred1 : array-like
            Predictions from first model
        y_pred2 : array-like
            Predictions from second model
        alpha : float
            Significance level (default: 0.05)
        correction : bool
            Whether to apply continuity correction (default: True)
            
        Returns:
        --------
        dict : Results of McNemar's test
        """
        # Create contingency table for McNemar's test
        # Both models correct, Both models wrong, Model1 correct Model2 wrong, Model1 wrong Model2 correct
        both_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
        both_wrong = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
        model1_correct_model2_wrong = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
        model1_wrong_model2_correct = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
        
        # Create 2x2 contingency table
        contingency_table = np.array([
            [both_correct, model1_correct_model2_wrong],
            [model1_wrong_model2_correct, both_wrong]
        ])
        
        # Perform McNemar's test
        if correction:
            # With continuity correction
            statistic = (abs(model1_correct_model2_wrong - model1_wrong_model2_correct) - 1)**2 / (model1_correct_model2_wrong + model1_wrong_model2_correct)
        else:
            # Without continuity correction
            statistic = (model1_correct_model2_wrong - model1_wrong_model2_correct)**2 / (model1_correct_model2_wrong + model1_wrong_model2_correct)
        
        # Calculate p-value
        p_value = 1 - chi2.cdf(statistic, df=1)
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Calculate effect size (Cohen's w)
        total = np.sum(contingency_table)
        expected = total / 4  # Expected frequency under null hypothesis
        effect_size = np.sqrt(np.sum((contingency_table - expected)**2 / expected) / total)
        
        results = {
            'test_name': 'McNemar\'s Test',
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': alpha,
            'contingency_table': contingency_table,
            'effect_size': effect_size,
            'interpretation': self._interpret_mcnemar(p_value, alpha, effect_size)
        }
        
        return results
    
    def _interpret_mcnemar(self, p_value, alpha, effect_size):
        """Interpret McNemar's test results"""
        if p_value < alpha:
            if effect_size < 0.1:
                return "Significant difference with small effect size"
            elif effect_size < 0.3:
                return "Significant difference with medium effect size"
            else:
                return "Significant difference with large effect size"
        else:
            return "No significant difference between models"
    
    def confusion_matrix_analysis(self, y_true, y_pred, model_name="Model"):
        """
        Comprehensive confusion matrix analysis
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        model_name : str
            Name of the model for labeling
            
        Returns:
        --------
        dict : Analysis results
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate Matthews Correlation Coefficient (MCC)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
        
        # Calculate Cohen's Kappa
        pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / ((tp + tn + fp + fn) ** 2)
        po = accuracy
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0
        
        results = {
            'model_name': model_name,
            'confusion_matrix': cm,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1_score,
                'mcc': mcc,
                'kappa': kappa
            },
            'raw_counts': {
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp
            }
        }
        
        return results
    
    def statistical_significance_test(self, y_true, y_pred1, y_pred2, test_type='mcnemar', alpha=0.05):
        """
        Perform statistical significance test between two models
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred1 : array-like
            Predictions from first model
        y_pred2 : array-like
            Predictions from second model
        test_type : str
            Type of test ('mcnemar', 'chi2')
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Test results
        """
        if test_type == 'mcnemar':
            return self.mcnemar_test(y_true, y_pred1, y_pred2, alpha)
        elif test_type == 'chi2':
            return self._chi2_test(y_true, y_pred1, y_pred2, alpha)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _chi2_test(self, y_true, y_pred1, y_pred2, alpha=0.05):
        """Perform Chi-square test for independence"""
        # Create contingency table
        table = np.array([
            [np.sum((y_pred1 == y_true) & (y_pred2 == y_true)), np.sum((y_pred1 == y_true) & (y_pred2 != y_true))],
            [np.sum((y_pred1 != y_true) & (y_pred2 == y_true)), np.sum((y_pred1 != y_true) & (y_pred2 != y_true))]
        ])
        
        chi2_stat, p_value, dof, expected = chi2_contingency(table)
        
        results = {
            'test_name': 'Chi-Square Test',
            'statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_significant': p_value < alpha,
            'alpha': alpha,
            'contingency_table': table
        }
        
        return results
    

    
    def regression_statistical_tests(self, y_true, y_pred1, y_pred2, alpha=0.05):
        """
        Perform statistical tests for regression models
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred1 : array-like
            Predictions from first model
        y_pred2 : array-like
            Predictions from second model
        alpha : float
            Significance level
            
        Returns:
        --------
        dict : Test results
        """
        # Calculate residuals
        residuals1 = y_true - y_pred1
        residuals2 = y_true - y_pred2
        
        # Paired t-test for residuals
        t_stat, p_value_t = ttest_rel(residuals1, residuals2)
        
        # Wilcoxon signed-rank test
        w_stat, p_value_w = wilcoxon(residuals1, residuals2)
        
        results = {
            'paired_t_test': {
                'statistic': t_stat,
                'p_value': p_value_t,
                'is_significant': p_value_t < alpha
            },
            'wilcoxon_test': {
                'statistic': w_stat,
                'p_value': p_value_w,
                'is_significant': p_value_w < alpha
            }
        }
        
        return results
    
    def model_comparison_summary(self, results_dict):
        """
        Create a comprehensive summary of model comparison results
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary containing results from various tests
            
        Returns:
        --------
        str : Formatted summary
        """
        summary = "=" * 80 + "\n"
        summary += "MODEL COMPARISON STATISTICAL ANALYSIS SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        for test_name, result in results_dict.items():
            summary += f"{test_name.upper()}\n"
            summary += "-" * len(test_name) + "\n"
            
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, dict):
                        summary += f"  {key}:\n"
                        for sub_key, sub_value in value.items():
                            summary += f"    {sub_key}: {sub_value}\n"
                    else:
                        summary += f"  {key}: {value}\n"
            else:
                summary += f"  Result: {result}\n"
            
            summary += "\n"
        
        return summary
    
    def plot_confusion_matrices(self, results_list, save_path=None):
        """
        Plot confusion matrices for multiple models
        
        Parameters:
        -----------
        results_list : list
            List of confusion matrix analysis results
        save_path : str, optional
            Path to save the plot
        """
        n_models = len(results_list)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, result in enumerate(results_list):
            cm = result['confusion_matrix']
            model_name = result['model_name']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, y_true, y_pred_dict, plot_type='roc', save_path=None):
        """
        Plot model comparison charts
        
        Parameters:
        -----------
        y_true : array-like
            True labels/values
        y_pred_dict : dict
            Dictionary of model predictions
        plot_type : str
            Type of plot ('roc', 'pr', 'residuals')
        save_path : str, optional
            Path to save the plot
        """
        if plot_type == 'roc':
            self._plot_roc_comparison(y_true, y_pred_dict, save_path)
        elif plot_type == 'pr':
            self._plot_pr_comparison(y_true, y_pred_dict, save_path)
        elif plot_type == 'residuals':
            self._plot_residual_comparison(y_true, y_pred_dict, save_path)
    
    def _plot_roc_comparison(self, y_true, y_pred_dict, save_path):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(8, 6))
        
        for model_name, y_pred in y_pred_dict.items():
            if len(np.unique(y_pred)) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc_score = roc_auc_score(y_true, y_pred)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_pr_comparison(self, y_true, y_pred_dict, save_path):
        """Plot Precision-Recall curves for multiple models"""
        plt.figure(figsize=(8, 6))
        
        for model_name, y_pred in y_pred_dict.items():
            if len(np.unique(y_pred)) == 2:  # Binary classification
                precision, recall, _ = precision_recall_curve(y_true, y_pred)
                ap_score = average_precision_score(y_true, y_pred)
                plt.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_residual_comparison(self, y_true, y_pred_dict, save_path):
        """Plot residual comparison for regression models"""
        n_models = len(y_pred_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            residuals = y_true - y_pred
            axes[i].scatter(y_pred, residuals, alpha=0.6)
            axes[i].axhline(y=0, color='r', linestyle='--')
            axes[i].set_xlabel('Predicted Values')
            axes[i].set_ylabel('Residuals')
            axes[i].set_title(f'{model_name} Residuals')
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def example_usage():
    """
    Example usage of the StatisticalAnalyzer class
    """
    # Create analyzer instance
    analyzer = StatisticalAnalyzer()
    
    # Example data (replace with your actual data)
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate true labels and predictions from two models
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred1 = np.random.binomial(1, 0.7, n_samples)  # Model 1 predictions
    y_pred2 = np.random.binomial(1, 0.6, n_samples)  # Model 2 predictions
    
    print("Example Statistical Analysis")
    print("=" * 50)
    
    # 1. McNemar's test
    print("\n1. McNemar's Test:")
    mcnemar_result = analyzer.mcnemar_test(y_true, y_pred1, y_pred2)
    print(f"Statistic: {mcnemar_result['statistic']:.4f}")
    print(f"P-value: {mcnemar_result['p_value']:.4f}")
    print(f"Significant: {mcnemar_result['is_significant']}")
    print(f"Interpretation: {mcnemar_result['interpretation']}")
    
    # 2. Confusion matrix analysis
    print("\n2. Confusion Matrix Analysis:")
    cm_result1 = analyzer.confusion_matrix_analysis(y_true, y_pred1, "Model 1")
    cm_result2 = analyzer.confusion_matrix_analysis(y_true, y_pred2, "Model 2")
    
    print(f"Model 1 Accuracy: {cm_result1['metrics']['accuracy']:.4f}")
    print(f"Model 2 Accuracy: {cm_result2['metrics']['accuracy']:.4f}")
    
    # 3. Statistical significance tests
    print("\n3. Statistical Significance Tests:")
    chi2_result = analyzer.statistical_significance_test(y_true, y_pred1, y_pred2, 'chi2')
    print(f"Chi-square test p-value: {chi2_result['p_value']:.4f}")
    
    # 4. Plot confusion matrices
    print("\n4. Generating plots...")
    analyzer.plot_confusion_matrices([cm_result1, cm_result2])
    
    # 5. Model comparison plots
    y_pred_dict = {'Model 1': y_pred1, 'Model 2': y_pred2}
    analyzer.plot_model_comparison(y_true, y_pred_dict, 'roc')
    
    # 6. Generate summary
    all_results = {
        'McNemar Test': mcnemar_result,
        'Chi-square Test': chi2_result,
        'Model 1 Analysis': cm_result1,
        'Model 2 Analysis': cm_result2
    }
    
    summary = analyzer.model_comparison_summary(all_results)
    print("\n" + summary)


if __name__ == "__main__":
    example_usage() 