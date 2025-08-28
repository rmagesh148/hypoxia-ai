#!/usr/bin/env python3
<<<<<<< Updated upstream
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 22:29:32 2025

@author: saiful
"""

#!/usr/bin/env python3
>>>>>>> Stashed changes
"""
Run significance tests on hypoxia prediction models using existing StatisticalAnalyzer
Loads pre-trained models and generates predictions without retraining
"""

import os
import numpy as np
import pandas as pd
import torch
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statistical_analysis import StatisticalAnalyzer
import warnings
warnings.filterwarnings('ignore')

<<<<<<< Updated upstream
=======
test_yr_mnth = "_year24_month_5-6-7-8th"

>>>>>>> Stashed changes
def load_data_and_prepare():
    """Load data and prepare for model inference"""
    
    print("Loading and preparing data...")
    
    # Load the preprocessed data
<<<<<<< Updated upstream
    df_scale_vector_rbf = pd.read_pickle('df_hyp_input.pkl')
=======
    # df_scale_vector_rbf = pd.read_pickle('df_hyp_input.pkl')
    df_scale_vector_rbf = pd.read_pickle('df_hyp_input_2018_2025.pkl')

>>>>>>> Stashed changes
    df_hyp = df_scale_vector_rbf.copy()
    
    # Create binary classification target
    df_hyp['oxy_class'] = np.where(df_hyp['oxyg'] > 2.0, 0, 1)
    print("Number of hypoxia cases:", df_hyp['oxy_class'].value_counts())
    
    # Import necessary functions from driver
<<<<<<< Updated upstream
    from driver import preprocess, prepare_dataset_2
=======
    from driver_test_a import preprocess, prepare_dataset_2
>>>>>>> Stashed changes
    
    # Preprocess data
    df_time = df_hyp[['ocean_date_time', 'oxyg']].copy()
    df_feat = preprocess(df_hyp)
    df_proc = pd.concat([df_time, df_feat], axis=1)
    df_proc['oxy_class'] = df_hyp['oxy_class']
    
    print(f"Processed DataFrame shape: {df_proc.shape}")
    
    # Prepare dataset
    X_train, y_train, X_test, y_test, *_ = prepare_dataset_2(
        df_proc.rename_axis('ocean_date').reset_index(), 
        predictor="oxy_class", lookback=7
    )
    
<<<<<<< Updated upstream
    if X_train is None or X_test is None:
        print("No valid data available. Exiting.")
        return None, None, None
=======
    # if X_train is None or X_test is None:
    #     print("No valid data available. Exiting.")
    #     return None, None, None
>>>>>>> Stashed changes
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_test, y_test, X_train.shape[2]

def load_trained_models(input_dim, output_dim=2):
    """Load pre-trained models without retraining"""
    
    print("Loading pre-trained models...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import model functions
    from models import LSTMClassifier, STT, Medformer, TCNClassifier
    
    models = {}
    model_files = {
<<<<<<< Updated upstream
        'lstm': 'lstm_model.pth',
        'stt': 'stt_model.pth', 
        'medformer': 'medformer_model.pth',
        'tcn': 'tcn_model.pth'
=======
        'lstm': 'saved_models/lstm_model_epoch30.pth',
        'stt': 'saved_models/stt_model_epoch30.pth', 
        'medformer': 'saved_models/medformer_model_epoch30.pth',
        'tcn': 'saved_models/tcn_model_epoch30.pth'
>>>>>>> Stashed changes
    }
    
    for model_name, model_file in model_files.items():
        if os.path.exists(model_file):
            print(f"Loading {model_name.upper()} model from {model_file}...")
            
            # Create model instance with exact parameters used in training
            if model_name == 'lstm':
                model = LSTMClassifier(input_dim, 120, 2, output_dim, dropout=0.3, bidirectional=True)
            elif model_name == 'stt':
                model = STT(input_dim, output_dim)
            elif model_name == 'medformer':
                model = Medformer(input_dim, output_dim)
            elif model_name == 'tcn':
                model = TCNClassifier(input_dim, output_dim, num_channels=[128,128,128], kernel_size=3, dropout=0.1)
            
            # Load trained weights
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            
            models[model_name] = model
            print(f"  {model_name.upper()} model loaded successfully")
        else:
            print(f"Warning: {model_file} not found, skipping {model_name}")
    
    return models, device

def generate_predictions(models, X_test, device):
    """Generate predictions from all loaded models"""
    
    print("Generating predictions from models...")
    
    predictions = {}
    
    with torch.no_grad():
        for model_name, model in models.items():
            print(f"Generating predictions for {model_name.upper()}...")
            
            # Convert to tensor if needed
            if not torch.is_tensor(X_test):
                X_test_tensor = torch.FloatTensor(X_test).to(device)
            else:
                X_test_tensor = X_test.to(device)
            
            # Generate predictions
            outputs = model(X_test_tensor)
            if outputs.shape[1] == 2:  # Binary classification
                y_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            else:  # Single output
                y_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            y_pred = (y_probs > 0.5).astype(int)
            
            predictions[model_name] = {
                'probabilities': y_probs,
                'predictions': y_pred
            }
            
            print(f"  {model_name.upper()} predictions generated")
    
    return predictions

def run_significance_tests(predictions, y_test):
    """Run significance tests between all model pairs"""
    
    analyzer = StatisticalAnalyzer()
    results = {}
    
    model_names = list(predictions.keys())
    print(f"Running significance tests on {len(model_names)} models...")
    
    # Test all model pairs
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:  # Avoid duplicates and self-comparisons
                print(f"\nTesting {model1.upper()} vs {model2.upper()}...")
                
                y_pred1 = predictions[model1]['predictions']
                y_pred2 = predictions[model2]['predictions']
                
                # McNemar's test
                mcnemar_result = analyzer.mcnemar_test(y_test, y_pred1, y_pred2)
                
                # Chi-square test
                chi2_result = analyzer.statistical_significance_test(y_test, y_pred1, y_pred2, 'chi2')
                
                # Confusion matrix analysis
                cm1 = analyzer.confusion_matrix_analysis(y_test, y_pred1, model1)
                cm2 = analyzer.confusion_matrix_analysis(y_test, y_pred2, model2)
                
                # Store results
                pair_name = f"{model1}_vs_{model2}"
                results[pair_name] = {
                    'mcnemar_test': mcnemar_result,
                    'chi2_test': chi2_result,
                    'model1_analysis': cm1,
                    'model2_analysis': cm2
                }
                
                print(f"  McNemar p-value: {mcnemar_result['p_value']:.2e}")
                print(f"  Chi-square p-value: {chi2_result['p_value']:.2e}")
                print(f"  Effect size: {mcnemar_result['effect_size']:.4f}")
    
    return results

def create_results_folder():
    """Create results folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
<<<<<<< Updated upstream
    results_dir = f"significance_test_results_{timestamp}"
=======
    results_dir = f"significance_test_results_{test_yr_mnth}"
>>>>>>> Stashed changes
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_results(results, predictions, y_test, results_dir):
    """Save all results to the results folder"""
    
    print(f"\nSaving results to {results_dir}/...")
    
    # 1. Save detailed results as JSON
    import json
    
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(results)
    
    with open(f'{results_dir}/significance_test_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # 2. Save summary report
<<<<<<< Updated upstream
    with open(f'{results_dir}/summary_report.txt', 'w') as f:
=======
    with open(f'{results_dir}/summary_report_{test_yr_mnth}.txt', 'w') as f:
>>>>>>> Stashed changes
        f.write("HYPOXIA MODEL SIGNIFICANCE TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for model_name in predictions.keys():
            y_pred = predictions[model_name]['predictions']
            accuracy = np.mean(y_pred == y_test)
            f.write(f"{model_name.upper()}: {accuracy:.4f}\n")
        
        f.write("\nSIGNIFICANCE TEST RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for pair_name, result in results.items():
            f.write(f"\n{pair_name.upper()}:\n")
            f.write(f"  McNemar Test: p-value = {result['mcnemar_test']['p_value']:.2e}\n")
            f.write(f"  Chi-Square Test: p-value = {result['chi2_test']['p_value']:.2e}\n")
            f.write(f"  Effect Size: {result['mcnemar_test']['effect_size']:.4f}\n")
            f.write(f"  Interpretation: {result['mcnemar_test']['interpretation']}\n")
    
    # 3. Create visualizations
    create_visualizations(results, predictions, y_test, results_dir)
    
    print(f"Results saved successfully to {results_dir}/")

def create_visualizations(results, predictions, y_test, results_dir):
    """Create visualization plots"""
    
    print("Creating visualizations...")
    
    # 1. Model accuracy comparison
    model_names = list(predictions.keys())
    accuracies = []
    
    for model_name in model_names:
        y_pred = predictions[model_name]['predictions']
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Hypoxia Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
<<<<<<< Updated upstream
    plt.savefig(f'{results_dir}/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
=======
    plt.savefig(f'{results_dir}/model_accuracy_comparison_{test_yr_mnth}.png', dpi=300, bbox_inches='tight')
>>>>>>> Stashed changes
    plt.close()
    
    # 2. Significance test heatmap
    model_pairs = list(results.keys())
    p_values_mcnemar = []
    
    for pair in model_pairs:
        p_values_mcnemar.append(results[pair]['mcnemar_test']['p_value'])
    
    # Create comparison matrix
    n_models = len(model_names)
    mcnemar_matrix = np.full((n_models, n_models), np.nan)
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j:
                pair_name = f"{model1}_vs_{model2}" if i < j else f"{model2}_vs_{model1}"
                if pair_name in results:
                    mcnemar_matrix[i, j] = results[pair_name]['mcnemar_test']['p_value']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(mcnemar_matrix, annot=True, fmt='.2e', cmap='RdYlBu_r', 
                xticklabels=model_names, yticklabels=model_names, 
                cbar_kws={'label': 'P-value'})
    plt.title('McNemar Test P-values', fontweight='bold')
    plt.xlabel('Model 2')
    plt.ylabel('Model 1')
    plt.tight_layout()
<<<<<<< Updated upstream
    plt.savefig(f'{results_dir}/significance_test_heatmap.png', dpi=300, bbox_inches='tight')
=======
    plt.savefig(f'{results_dir}/significance_test_heatmap_{test_yr_mnth}.png', dpi=300, bbox_inches='tight')
>>>>>>> Stashed changes
    plt.close()
    
    # 3. Effect size comparison
    effect_sizes = []
    test_names = []
    
    for pair in model_pairs:
        effect_sizes.append(results[pair]['mcnemar_test']['effect_size'])
        test_names.append(pair.replace('_', ' vs ').upper())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(test_names, effect_sizes, color='lightsteelblue')
    plt.title('Effect Sizes from McNemar Tests', fontsize=14, fontweight='bold')
    plt.ylabel('Effect Size (Cohen\'s w)', fontsize=12)
    plt.xlabel('Model Comparison', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add effect size interpretation
    for bar, effect_size in zip(bars, effect_sizes):
        if effect_size < 0.1:
            interpretation = 'Small'
            color = 'green'
        elif effect_size < 0.3:
            interpretation = 'Medium'
            color = 'orange'
        else:
            interpretation = 'Large'
            color = 'red'
        
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{effect_size:.3f}\n({interpretation})', ha='center', va='bottom', 
                fontweight='bold', color=color)
    
    plt.tight_layout()
<<<<<<< Updated upstream
    plt.savefig(f'{results_dir}/effect_size_comparison.png', dpi=300, bbox_inches='tight')
=======
    plt.savefig(f'{results_dir}/effect_size_comparison_{test_yr_mnth}.png', dpi=300, bbox_inches='tight')
>>>>>>> Stashed changes
    plt.close()

def main():
    """Main function to run significance tests"""
    
    print("=" * 60)
    print("HYPOXIA MODEL SIGNIFICANCE TESTING")
    print("=" * 60)
    print("Loading pre-trained models (no retraining required)...")
    
    # Load data and prepare
    data_result = load_data_and_prepare()
    if data_result is None:
        return
    
    X_test, y_test, input_dim = data_result
    
    # Load trained models
    models, device = load_trained_models(input_dim)
    if not models:
        print("No models loaded. Exiting.")
        return
    
    print(f"Loaded {len(models)} models successfully")
    
    # Generate predictions
    predictions = generate_predictions(models, X_test, device)
    
    # Convert tensors to numpy arrays for statistical analysis
    if torch.is_tensor(y_test):
        y_test = y_test.cpu().numpy()
    
    # Also convert predictions to numpy arrays
    for model_name in predictions:
        if torch.is_tensor(predictions[model_name]['predictions']):
            predictions[model_name]['predictions'] = predictions[model_name]['predictions'].cpu().numpy()
        if torch.is_tensor(predictions[model_name]['probabilities']):
            predictions[model_name]['probabilities'] = predictions[model_name]['probabilities'].cpu().numpy()
    
    # Run significance tests
    results = run_significance_tests(predictions, y_test)
    
    # Create results folder and save everything
    results_dir = create_results_folder()
    save_results(results, predictions, y_test, results_dir)
    
    print("\n" + "=" * 60)
    print("SIGNIFICANCE TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved in: {results_dir}/")
    print("Files generated:")
    print("  - significance_test_results.json")
    print("  - summary_report.txt")
    print("  - model_accuracy_comparison.png")
    print("  - significance_test_heatmap.png")
    print("  - effect_size_comparison.png")

if __name__ == "__main__":
<<<<<<< Updated upstream
    main() 
=======
    main() 
    print('Execution Finished..')
>>>>>>> Stashed changes
