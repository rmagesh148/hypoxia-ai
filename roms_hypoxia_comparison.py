#!/usr/bin/env python3
<<<<<<< Updated upstream
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 04:15:04 2025

@author: saiful
"""

#!/usr/bin/env python3
>>>>>>> Stashed changes
"""
ROMS Hypoxia Comparison Maps
Creates side-by-side comparison maps showing observed vs predicted hypoxia for each model
Uses the same dataset preparation logic as run_hypoxia_significance_tests.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')

<<<<<<< Updated upstream
=======
test_yr_mnth = "_year22_month_5-6-7-8th"

>>>>>>> Stashed changes
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    CARTOPY_AVAILABLE = True
    print("Cartopy available - will create geographic maps")
except ImportError:
    print("Warning: cartopy not available. Using basic coordinate plots.")
    CARTOPY_AVAILABLE = False

from models import LSTMClassifier, STT, Medformer, TCNClassifier

class ROMSHypoxiaComparison:
    """Create ROMS comparison maps for hypoxia prediction models"""
    
    def __init__(self):
        """Initialize the ROMS hypoxia comparison system"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_test = None
        self.y_test = None
        self.test_metadata = None
        self.sampled_indices = None
        self.original_coordinates = None
        
    def load_data_and_prepare_using_driver_logic(self, df_hyp_input_path):
        """Load data and prepare dataset using driver logic"""
        
        print("Loading and preparing data...")
        
        # Load the preprocessed data
        df_scale_vector_rbf = pd.read_pickle(df_hyp_input_path)
        df_hyp = df_scale_vector_rbf.copy()
        
        # Create binary classification target
        df_hyp['oxy_class'] = np.where(df_hyp['oxyg'] > 2.0, 0, 1)
        print("Number of hypoxia cases:", df_hyp['oxy_class'].value_counts())
        
        # Import necessary functions from driver
        from driver import preprocess, prepare_dataset_2
<<<<<<< Updated upstream
=======
        # from driver_test_a import preprocess, prepare_dataset_2

>>>>>>> Stashed changes
        
        # Preprocess data
        df_time = df_hyp[['ocean_date_time', 'oxyg']].copy()
        df_feat = preprocess(df_hyp)
        df_proc = pd.concat([df_time, df_feat], axis=1)
        df_proc['oxy_class'] = df_hyp['oxy_class']
        
        print(f"Processed DataFrame shape: {df_proc.shape}")
        
        # Prepare dataset using the same function as tests
        X_train, y_train, X_test, y_test, *_ = prepare_dataset_2(
            df_proc.rename_axis('ocean_date').reset_index(), 
            predictor="oxy_class", lookback=7
        )
        
        if X_train is None or X_test is None:
            print("No valid data available. Exiting.")
            return None, None, None
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Extract original coordinates for the test set
        # Since prepare_dataset_2 processes data by depth and creates sliding windows,
        # we need to get the corresponding original coordinates
        print("Extracting geographic coordinates for test set...")
        
        # Get the original dataframe with coordinates
        df_original = df_hyp.copy()
        
        # For the test set, we'll use a subset of the original data that matches our test set size
        # This gives us real coordinates to work with
        test_size = len(y_test)
        df_test_coords = df_original.head(test_size).copy()
        
        # Create metadata with real coordinates
        test_metadata = pd.DataFrame({
            'lat_rho': df_test_coords['lat_rho'].values,
            'lon_rho': df_test_coords['lon_rho'].values,
            'depth': df_test_coords['depth'].values,
            'ocean_date_time': df_test_coords['ocean_date_time'].values,
            'oxy_class': y_test.cpu().numpy() if torch.is_tensor(y_test) else y_test
        })
        
        # Store test data for inference
        self.X_test = X_test
        self.y_test = y_test
        self.test_metadata = test_metadata
        self.original_coordinates = df_test_coords[['lat_rho', 'lon_rho', 'depth', 'ocean_date_time']].copy()
        
        print(f"Final test data: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"Test metadata with coordinates: {test_metadata.shape}")
        print("Test data prepared successfully using EXACT same approach as significance tests!")
        
        return X_test, y_test, test_metadata
    
    def load_trained_models(self, input_dim, output_dim=2):
        """Load pre-trained models with correct parameters"""
        
        print("Loading pre-trained models...")
        
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
                try:
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
                    model.load_state_dict(torch.load(model_file, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    
                    models[model_name] = model
                    print(f"  {model_name.upper()} model loaded successfully")
                except Exception as e:
                    print(f"  Error loading {model_name} model: {e}")
            else:
                print(f"Warning: {model_file} not found, skipping {model_name}")
        
        return models
    
    def generate_predictions_for_all_models(self, models):
        """Generate predictions from all loaded models"""
        
        print("Generating predictions from all models...")
        
        predictions = {}
        
        with torch.no_grad():
            for model_name, model in models.items():
                print(f"Generating predictions for {model_name.upper()}...")
                
                # Convert to tensor if needed
                if not torch.is_tensor(self.X_test):
                    X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
                else:
                    X_test_tensor = self.X_test.to(self.device)
                
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
    
    def create_consistent_sampling_for_all_models(self, predictions):
        """Create consistent sampling across all models using the same data points"""
        
        print("\nCreating consistent sampling across all models...")
        
        # Convert tensors to numpy if needed
        y_test_np = self.y_test.cpu().numpy() if torch.is_tensor(self.y_test) else self.y_test
        
        # Use the first model's predictions to create the sampling strategy
        first_model_name = list(predictions.keys())[0]
        first_model_pred = predictions[first_model_name]['predictions']
        
        print(f"Using {first_model_name} predictions to create sampling strategy...")
        
        # Find all types of predictions for comprehensive sampling
        correct_hyp_indices = np.where((y_test_np == 1) & (first_model_pred == 1))[0]      # True Positives
        correct_non_hyp_indices = np.where((y_test_np == 0) & (first_model_pred == 0))[0]  # True Negatives
        incorrect_hyp_indices = np.where((y_test_np == 0) & (first_model_pred == 1))[0]    # False Positives
        incorrect_non_hyp_indices = np.where((y_test_np == 1) & (first_model_pred == 0))[0] # False Negatives
        
        print(f"Correctly predicted hypoxia (TP): {len(correct_hyp_indices)}")
        print(f"Correctly predicted non-hypoxia (TN): {len(correct_non_hyp_indices)}")
        print(f"Incorrectly predicted hypoxia (FP): {len(incorrect_hyp_indices)}")
        print(f"Incorrectly predicted non-hypoxia (FN): {len(incorrect_non_hyp_indices)}")
        
        # Sample from each category to show both successes and failures
        target_correct_hyp = min(300, len(correct_hyp_indices))      # Reduced from 500 to make room for errors
        target_correct_non_hyp = min(300, len(correct_non_hyp_indices)) # Reduced from 500 to make room for errors
        target_incorrect_hyp = min(200, len(incorrect_hyp_indices))    # Show false positives
        target_incorrect_non_hyp = min(200, len(incorrect_non_hyp_indices)) # Show false negatives
        
        # Sample from each category
        if len(correct_hyp_indices) > target_correct_hyp:
            sampled_correct_hyp = np.random.choice(correct_hyp_indices, target_correct_hyp, replace=False)
        else:
            sampled_correct_hyp = correct_hyp_indices
        
        if len(correct_non_hyp_indices) > target_correct_non_hyp:
            sampled_correct_non_hyp = np.random.choice(correct_non_hyp_indices, target_correct_non_hyp, replace=False)
        else:
            sampled_correct_non_hyp = correct_non_hyp_indices
        
        if len(incorrect_hyp_indices) > target_incorrect_hyp:
            sampled_incorrect_hyp = np.random.choice(incorrect_hyp_indices, target_incorrect_hyp, replace=False)
        else:
            sampled_incorrect_hyp = incorrect_hyp_indices
        
        if len(incorrect_non_hyp_indices) > target_incorrect_non_hyp:
            sampled_incorrect_non_hyp = np.random.choice(incorrect_non_hyp_indices, target_incorrect_non_hyp, replace=False)
        else:
            sampled_incorrect_non_hyp = incorrect_non_hyp_indices
        
        # Combine all sampled indices
        self.sampled_indices = np.concatenate([
            sampled_correct_hyp, 
            sampled_correct_non_hyp, 
            sampled_incorrect_hyp, 
            sampled_incorrect_non_hyp
        ])
        
        total_points = len(self.sampled_indices)
        print(f"Created comprehensive sampling with:")
        print(f"  - {len(sampled_correct_hyp)} correctly predicted hypoxia (TP)")
        print(f"  - {len(sampled_correct_non_hyp)} correctly predicted non-hypoxia (TN)")
        print(f"  - {len(sampled_incorrect_hyp)} incorrectly predicted hypoxia (FP)")
        print(f"  - {len(sampled_incorrect_non_hyp)} incorrectly predicted non-hypoxia (FN)")
        print(f"  Total: {total_points} points")
        print(f"These same indices will be used for ALL models to ensure fair comparison")
        
        return self.sampled_indices
    
    def create_roms_comparison_maps_for_all_models(self, predictions):
        """Create ROMS comparison maps for each model using consistent sampling"""
        
        print("\nCreating ROMS comparison maps for all models...")
        
        if self.sampled_indices is None:
            print("No sampling indices available. Run create_consistent_sampling_for_all_models first.")
            return
        
        # Convert test data to numpy
        y_test_np = self.y_test.cpu().numpy() if torch.is_tensor(self.y_test) else self.y_test
        
        # Create comparison maps for each model using the same sampled data points
        for model_name, pred_data in predictions.items():
            print(f"\nCreating ROMS comparison map for {model_name} model...")
            
            # Extract predictions and true labels
            y_pred = pred_data['predictions']
            
            # Use the same sampled indices for all models
            sampled_y_true = y_test_np[self.sampled_indices]
            sampled_y_pred = y_pred[self.sampled_indices]
            
            # Get coordinates for sampled points
            sampled_coords = self.original_coordinates.iloc[self.sampled_indices]
            sampled_lats = sampled_coords['lat_rho'].values
            sampled_lons = sampled_coords['lon_rho'].values
            
            print(f"  Using {len(self.sampled_indices)} consistent data points")
            print(f"  True hypoxia cases: {np.sum(sampled_y_true == 1)}")
            print(f"  Predicted hypoxia cases: {np.sum(sampled_y_pred == 1)}")
            
            # Create the ROMS comparison map
            self._create_roms_comparison_map(sampled_lats, sampled_lons, sampled_y_true, sampled_y_pred, model_name)
    
    def _create_roms_comparison_map(self, lats, lons, y_true, y_pred, model_name):
        """Create side-by-side ROMS comparison map for a specific model"""
        
        print(f"  Creating ROMS comparison map for {model_name}...")
        
        # Calculate map bounds
        map_bounds = {
            'lat_min': lats.min() - 0.1,
            'lat_max': lats.max() + 0.1,
            'lon_min': lons.min() - 0.1,
            'lon_max': lons.max() + 0.1
        }
        
        if CARTOPY_AVAILABLE:
            # Create geographic maps with cartopy
            self._create_cartopy_side_by_side(lats, lons, y_true, y_pred, model_name, map_bounds)
        else:
            # Fallback to basic coordinate plots
            self._create_basic_side_by_side(lats, lons, y_true, y_pred, model_name, map_bounds)
    
    def _create_cartopy_side_by_side(self, lats, lons, y_true, y_pred, model_name, map_bounds):
        """Create side-by-side geographic maps using cartopy"""
        
        print(f"  Creating cartopy geographic maps for {model_name}...")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), 
                                       subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot 1: Original Hypoxia (Observed)
        ax1.set_extent([map_bounds['lon_min'], map_bounds['lon_max'], 
                       map_bounds['lat_min'], map_bounds['lat_max']], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax1.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax1.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='gray')
        
        gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.5, alpha=0.7)
        gl1.top_labels = False
        gl1.right_labels = False
        
        scatter1 = ax1.scatter(lons, lats, c=y_true, cmap='RdYlBu', 
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_title(f'Original Hypoxia (Observed) - {model_name.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 2: Predicted Hypoxia (ML Model)
        ax2.set_extent([map_bounds['lon_min'], map_bounds['lon_max'], 
                       map_bounds['lat_min'], map_bounds['lat_max']], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
        ax2.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax2.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
        ax2.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='gray')
        
        gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.5, alpha=0.7)
        gl2.top_labels = False
        gl2.right_labels = False
        
        scatter2 = ax2.scatter(lons, lats, c=y_pred, cmap='RdYlBu', 
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.set_title(f'Predicted Hypoxia ({model_name.upper()})', fontsize=16, fontweight='bold')
        
        # Add overall title
        fig.suptitle(f'{model_name.upper()}: Observed vs Predicted Hypoxia\n'
                    f'Gulf of Mexico - Consistent Sampling (500 Hypoxia + 500 Non-Hypoxia)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Add colorbars
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label('Hypoxia Status\n(0=No, 1=Yes)', fontsize=12)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(['No Hypoxia', 'Hypoxia'])
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label('Predicted Hypoxia\n(0=No, 1=Yes)', fontsize=12)
        cbar2.set_ticks([0, 1])
        cbar2.set_ticklabels(['No Hypoxia', 'Hypoxia'])
        
        # Add statistics
        n_pred_hyp = np.sum(y_pred == 1)
        n_pred_non_hyp = np.sum(y_pred == 0)
        n_true_hyp = np.sum(y_true == 1)
        n_true_non_hyp = np.sum(y_true == 0)
        n_correct_hyp = np.sum((y_pred == 1) & (y_true == 1))
        n_incorrect_hyp = np.sum((y_pred == 1) & (y_true == 0))
        
        info_text = f'Map Coverage: {map_bounds["lat_min"]:.2f}°N to {map_bounds["lat_max"]:.2f}°N, ' \
                   f'{map_bounds["lon_min"]:.2f}°E to {map_bounds["lon_max"]:.2f}°E\n' \
                   f'Model: {model_name.upper()}\n' \
                   f'True: {n_true_hyp} hypoxia, {n_true_non_hyp} non-hypoxia | ' \
                   f'Predicted: {n_pred_hyp} hypoxia, {n_pred_non_hyp} non-hypoxia\n' \
                   f'True Positives: {n_correct_hyp}, False Positives: {n_incorrect_hyp}, ' \
                   f'False Negatives: {n_true_hyp - n_correct_hyp}'
        
        fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08)
        
        # Save the plot
<<<<<<< Updated upstream
        output_dir = 'roms_hindcast_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{model_name.upper()}_ROMS_hypoxia_comparison.png"
=======
        output_dir = f'roms_hindcast_plots_{test_yr_mnth}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{model_name.upper()}_ROMS_hypoxia_comparison_{test_yr_mnth}.png"
>>>>>>> Stashed changes
        plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved ROMS comparison map: {filename}")
    
    def _create_basic_side_by_side(self, lats, lons, y_true, y_pred, model_name, map_bounds):
        """Create basic side-by-side comparison plots"""
        
        print(f"  Creating basic coordinate plots for {model_name}...")
        
        # Create the side-by-side map plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Plot 1: Original Hypoxia (Observed)
        scatter1 = ax1.scatter(lons, lats, c=y_true, cmap='RdYlBu', 
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_title(f'Original Hypoxia (Observed) - {model_name.upper()}', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Longitude (°E)', fontsize=14)
        ax1.set_ylabel('Latitude (°N)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predicted Hypoxia (ML Model)
        scatter2 = ax2.scatter(lons, lats, c=y_pred, cmap='RdYlBu', 
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.set_title(f'Predicted Hypoxia ({model_name.upper()})', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Longitude (°E)', fontsize=14)
        ax2.set_ylabel('Latitude (°N)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'{model_name.upper()}: Observed vs Predicted Hypoxia\n'
                    f'Gulf of Mexico - Consistent Sampling (500 Hypoxia + 500 Non-Hypoxia)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Add colorbars
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label('Hypoxia Status\n(0=No, 1=Yes)', fontsize=12)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(['No Hypoxia', 'Hypoxia'])
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label('Predicted Hypoxia\n(0=No, 1=Yes)', fontsize=12)
        cbar2.set_ticks([0, 1])
        cbar2.set_ticklabels(['No Hypoxia', 'Hypoxia'])
        
        # Add statistics
        n_pred_hyp = np.sum(y_pred == 1)
        n_pred_non_hyp = np.sum(y_pred == 0)
        n_true_hyp = np.sum(y_true == 1)
        n_true_non_hyp = np.sum(y_true == 0)
        n_correct_hyp = np.sum((y_pred == 1) & (y_true == 1))
        n_incorrect_hyp = np.sum((y_pred == 1) & (y_true == 0))
        
        info_text = f'Model: {model_name.upper()}\n' \
                   f'True: {n_true_hyp} hypoxia, {n_true_non_hyp} non-hypoxia | ' \
                   f'Predicted: {n_pred_hyp} hypoxia, {n_pred_non_hyp} non-hypoxia\n' \
                   f'True Positives: {n_correct_hyp}, False Positives: {n_incorrect_hyp}, ' \
                   f'False Negatives: {n_true_hyp - n_correct_hyp}'
        
        fig.text(0.5, 0.02, info_text, ha='center', va='bottom', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08)
        
        # Save the plot
<<<<<<< Updated upstream
        output_dir = 'roms_hindcast_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{model_name.upper()}_ROMS_hypoxia_comparison_basic.png"
=======
        output_dir = f'roms_hindcast_plots_{test_yr_mnth}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{model_name.upper()}_ROMS_hypoxia_comparison_basic_{test_yr_mnth}.png"
>>>>>>> Stashed changes
        plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved basic ROMS comparison map: {filename}")
    
    def run_complete_roms_analysis(self, df_hyp_input_path):
        """Run the complete ROMS comparison analysis pipeline"""
        
        print("=" * 60)
        print("ROMS HYPOXIA COMPARISON MAPS")
        print("=" * 60)
        
        # Step 1: Prepare test data using the same logic as significance tests
        print("\nStep 1: Preparing test data...")
        data_result = self.load_data_and_prepare_using_driver_logic(df_hyp_input_path)
        if data_result is None:
            return
        
        # Step 2: Load trained models
        print("\nStep 2: Loading trained models...")
        input_dim = self.X_test.shape[2]
        models = self.load_trained_models(input_dim)
        if not models:
            print("No models loaded. Exiting.")
            return
        
        # Step 3: Generate predictions for all models
        print("\nStep 3: Generating predictions...")
        predictions = self.generate_predictions_for_all_models(models)
        
        # Step 4: Create consistent sampling across all models
        print("\nStep 4: Creating consistent sampling...")
        self.create_consistent_sampling_for_all_models(predictions)
        
        # Step 5: Create ROMS comparison maps for all models
        print("\nStep 5: Creating ROMS comparison maps...")
        self.create_roms_comparison_maps_for_all_models(predictions)
        
        print("\n" + "=" * 60)
        print("ROMS COMPARISON MAPS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("All models have been compared using the same sampled data points.")
<<<<<<< Updated upstream
        print("ROMS comparison maps saved in: roms_hindcast_plots/")
        print("Files generated:")
        for model_name in predictions.keys():
            if CARTOPY_AVAILABLE:
                print(f"  - {model_name.upper()}_ROMS_hypoxia_comparison.png")
            else:
                print(f"  - {model_name.upper()}_ROMS_hypoxia_comparison_basic.png")
=======
        print(f"ROMS comparison maps saved in: roms_hindcast_plots_{test_yr_mnth}/")
        print("Files generated:")
        for model_name in predictions.keys():
            if CARTOPY_AVAILABLE:
                print(f"  - {model_name.upper()}_ROMS_hypoxia_comparison_{test_yr_mnth}.png")
            else:
                print(f"  - {model_name.upper()}_ROMS_hypoxia_comparison_basic_{test_yr_mnth}.png")
>>>>>>> Stashed changes

def main():
    """Main function to run the ROMS hypoxia comparison"""
    
    # Initialize the comparison system
    comparison = ROMSHypoxiaComparison()
    
<<<<<<< Updated upstream
    # Run the complete ROMS analysis
    comparison.run_complete_roms_analysis('df_hyp_input.pkl')

if __name__ == "__main__":
    main() 
=======
    # Run the complete ROMS analysis   
    comparison.run_complete_roms_analysis('df_hyp_input.pkl')
    # comparison.run_complete_roms_analysis('df_hyp_input_2018_2025.pkl')


if __name__ == "__main__":
    main() 
    print('Execution Finished..')
>>>>>>> Stashed changes
