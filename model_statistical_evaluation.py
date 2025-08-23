import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from statistical_analysis import StatisticalAnalyzer
import os

def load_model_predictions(model_paths, X_test, y_test, device):
    """
    Load trained models and generate predictions for statistical analysis
    
    Parameters:
    -----------
    model_paths : dict
        Dictionary mapping model names to their saved paths
    X_test : torch.Tensor
        Test features
    y_test : torch.Tensor
        Test labels
    device : torch.device
        Device to run models on
        
    Returns:
    --------
    dict : Dictionary of model predictions
    """
    predictions = {}
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"Loading {model_name} from {model_path}")
            
            # Import the appropriate model class based on model name
            if model_name == 'lstm':
                from models import LSTMClassifier
                model = LSTMClassifier(X_test.shape[2], 120, 2, 2, dropout=0.3, bidirectional=True)
            elif model_name == 'stt':
                from models import STT
                model = STT(X_test.shape[2], 2)
            elif model_name == 'medformer':
                from models import Medformer
                model = Medformer(X_test.shape[2], 2)
            elif model_name == 'tcn':
                from models import TCNClassifier
                model = TCNClassifier(X_test.shape[2], 2, num_channels=[128,128,128], kernel_size=3, dropout=0.1)
            else:
                print(f"Unknown model type: {model_name}")
                continue
            
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            # Generate predictions
            with torch.no_grad():
                logits = model(X_test.to(device))
                y_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_pred = (y_probs >= 0.5).astype(int)
                
                predictions[model_name] = {
                    'probabilities': y_probs,
                    'predictions': y_pred,
                    'logits': logits.cpu().numpy()
                }
                
            print(f"Generated predictions for {model_name}")
        else:
            print(f"Model file not found: {model_path}")
    
    return predictions

def comprehensive_model_evaluation(y_true, model_predictions, save_dir="statistical_results"):
    """
    Perform comprehensive statistical evaluation of multiple models
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    model_predictions : dict
        Dictionary of model predictions
    save_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize statistical analyzer
    analyzer = StatisticalAnalyzer()
    
    # Convert y_true to numpy if it's a tensor
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    
    # Extract model names and predictions
    model_names = list(model_predictions.keys())
    y_pred_dict = {name: model_predictions[name]['predictions'] for name in model_names}
    y_prob_dict = {name: model_predictions[name]['probabilities'] for name in model_names}
    
    print(f"Evaluating {len(model_names)} models: {model_names}")
    
    # 1. Individual model analysis
    print("\n1. Individual Model Analysis")
    print("=" * 50)
    
    individual_results = {}
    for model_name in model_names:
        print(f"\nAnalyzing {model_name}...")
        
        # Confusion matrix analysis
        cm_result = analyzer.confusion_matrix_analysis(
            y_true, 
            y_pred_dict[model_name], 
            model_name
        )
        
        # Calculate additional metrics
        auroc = roc_auc_score(y_true, y_prob_dict[model_name])
        aupr = average_precision_score(y_true, y_prob_dict[model_name])
        
        individual_results[model_name] = {
            'confusion_matrix': cm_result,
            'auroc': auroc,
            'aupr': aupr
        }
        
        print(f"  Accuracy: {cm_result['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {cm_result['metrics']['f1_score']:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  AUPR: {aupr:.4f}")
    
    # 2. Model comparison using McNemar's test
    print("\n2. Model Comparison with McNemar's Test")
    print("=" * 50)
    
    comparison_results = {}
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            print(f"\nComparing {model1} vs {model2}...")
            
            # McNemar's test
            mcnemar_result = analyzer.mcnemar_test(
                y_true, 
                y_pred_dict[model1], 
                y_pred_dict[model2]
            )
            
            comparison_results[f"{model1}_vs_{model2}"] = mcnemar_result
            
            print(f"  McNemar's Test p-value: {mcnemar_result['p_value']:.6f}")
            print(f"  Significant difference: {mcnemar_result['is_significant']}")
            print(f"  Effect size: {mcnemar_result['effect_size']:.4f}")
            print(f"  Interpretation: {mcnemar_result['interpretation']}")
    
    # 3. Additional statistical tests
    print("\n3. Additional Statistical Tests")
    print("=" * 50)
    
    additional_tests = {}
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            print(f"\nAdditional tests for {model1} vs {model2}...")
            
            # Chi-square test
            chi2_result = analyzer.statistical_significance_test(
                y_true, 
                y_pred_dict[model1], 
                y_pred_dict[model2], 
                'chi2'
            )
            
            additional_tests[f"{model1}_vs_{model2}"] = {
                'chi2': chi2_result
            }
            
            print(f"  Chi-square p-value: {chi2_result['p_value']:.6f}")
    
    # 4. Generate visualizations
    print("\n4. Generating Visualizations")
    print("=" * 50)
    
    # Confusion matrices
    cm_results = [individual_results[name]['confusion_matrix'] for name in model_names]
    analyzer.plot_confusion_matrices(
        cm_results, 
        save_path=os.path.join(save_dir, 'confusion_matrices.png')
    )
    
    # ROC curves
    analyzer.plot_model_comparison(
        y_true, 
        y_prob_dict, 
        'roc', 
        save_path=os.path.join(save_dir, 'roc_comparison.png')
    )
    
    # Precision-Recall curves
    analyzer.plot_model_comparison(
        y_true, 
        y_prob_dict, 
        'pr', 
        save_path=os.path.join(save_dir, 'pr_comparison.png')
    )
    
    # 5. Generate comprehensive summary
    print("\n5. Generating Summary Report")
    print("=" * 50)
    
    all_results = {
        'Individual Model Analysis': individual_results,
        'McNemar Test Results': comparison_results,
        'Additional Statistical Tests': additional_tests
    }
    
    summary = analyzer.model_comparison_summary(all_results)
    
    # Save summary to file
    summary_path = os.path.join(save_dir, 'statistical_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"Summary saved to: {summary_path}")
    
    # 6. Create performance comparison table
    print("\n6. Performance Comparison Table")
    print("=" * 50)
    
    performance_data = []
    for model_name in model_names:
        cm_result = individual_results[model_name]['confusion_matrix']
        performance_data.append({
            'Model': model_name,
            'Accuracy': f"{cm_result['metrics']['accuracy']:.4f}",
            'Precision': f"{cm_result['metrics']['precision']:.4f}",
            'Recall': f"{cm_result['metrics']['recall']:.4f}",
            'F1-Score': f"{cm_result['metrics']['f1_score']:.4f}",
            'MCC': f"{cm_result['metrics']['mcc']:.4f}",
            'AUROC': f"{individual_results[model_name]['auroc']:.4f}",
            'AUPR': f"{individual_results[model_name]['aupr']:.4f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    print(performance_df.to_string(index=False))
    
    # Save performance table
    performance_path = os.path.join(save_dir, 'performance_comparison.csv')
    performance_df.to_csv(performance_path, index=False)
    print(f"Performance table saved to: {performance_path}")
    
    return {
        'individual_results': individual_results,
        'comparison_results': comparison_results,
        'additional_tests': additional_tests,
        'performance_df': performance_df
    }

def run_hypoxia_model_evaluation():
    """
    Main function to run statistical evaluation on hypoxia models
    """
    print("Hypoxia Model Statistical Evaluation")
    print("=" * 50)
    
    # Check for available model files
    model_paths = {
        'lstm': 'lstm_model.pth',
        'stt': 'stt_model.pth', 
        'medformer': 'medformer_model.pth',
        'tcn': 'tcn_model.pth'
    }
    
    available_models = {k: v for k, v in model_paths.items() if os.path.exists(v)}
    
    if not available_models:
        print("No trained model files found. Please train models first using driver.py")
        return
    
    print(f"Found {len(available_models)} trained models: {list(available_models.keys())}")
    
    # Load test data (you'll need to modify this based on your data loading)
    try:
        # Try to load from pickle file
        df_proc = pd.read_pickle('df_hyp_input.pkl')
        print("Loaded data from pickle file")
        
        # Create binary classification target
        df_proc['oxy_class'] = np.where(df_proc['oxyg'] > 2.0, 0, 1)
        
        # Prepare dataset (using the same logic as in driver.py)
        from driver import prepare_dataset_2
        
        X_train, y_train, X_test, y_test = prepare_dataset_2(
            df_proc.rename_axis('ocean_date').reset_index(), 
            predictor="oxy_class", 
            lookback=7
        )
        
        if X_test is None:
            print("No test data available. Please check your data preparation.")
            return
            
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have the required data files and run driver.py first")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model predictions
    model_predictions = load_model_predictions(available_models, X_test, y_test, device)
    
    if not model_predictions:
        print("No model predictions generated. Exiting.")
        return
    
    # Run comprehensive evaluation
    results = comprehensive_model_evaluation(y_test, model_predictions)
    
    print("\nEvaluation completed successfully!")
    print("Check the 'statistical_results' directory for detailed results and visualizations.")

if __name__ == "__main__":
    run_hypoxia_model_evaluation() 