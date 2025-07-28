# Complete Implementation Guide for Gait Waveform Classification
# This script shows how to use the neural network with your specific data

# ===========================
# STEP 1: COMPLETE YOUR DATA PREPARATION
# ===========================

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Your existing data preparation code
df = pd.read_csv("dat_nn.csv")

# Check that you have exactly 100 items per subject per signal
assert df.groupby(['subject', 'signal_components']).size().nunique() == 1, "Inconsistent item count"

# Create a time-indexed variable name: X_1, X_2, ..., Y_101
df['feature_name'] = df['signal_components'] + "_" + df['item'].astype(str)

# Pivot: one row per subject, columns = feature_name
df_wide = df.pivot(index='subject', columns='feature_name', values='value').reset_index()

# Merge in group label (for classification)
labels = df[['subject', 'group']].drop_duplicates()
df_final = df_wide.merge(labels, on='subject')

# Split into features and labels 
X = df_final.drop(columns=['subject', 'group']).to_numpy()  # shape (n_subjects, 200)
y = df_final['group'].to_numpy()  # shape (n_subjects,)

# Split into training and test sets by subject 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Explore Dataset
m_train = X_train.shape[0]
num_features = X_train.shape[1]
m_test = X_test.shape[0]

print("=== Dataset Overview ===")
print(f"Number of training examples: {m_train}")
print(f"Number of testing examples: {m_test}")
print(f"Number of features per subject: {num_features}")
print(f"Train X shape: {X_train.shape}")
print(f"Train y shape: {y_train.shape}")
print(f"Test X shape: {X_test.shape}")
print(f"Test y shape: {y_test.shape}")
print(f"Unique labels: {np.unique(y)}")
print(f"Label distribution in training: {np.bincount(y_train == np.unique(y)[1])}")

# ===========================
# STEP 2: IMPORT THE NEURAL NETWORK CODE
# ===========================

# Here you would copy the complete neural network code from the first artifact
# or import it if you've saved it as a separate module

# ===========================
# STEP 3: TRAIN THE MODEL
# ===========================

def run_gait_classification_pipeline():
    """Complete pipeline for gait classification"""
    
    print("\n=== Starting Gait Waveform Classification ===")
    
    # Define network architecture
    # Input: 200 features (100 frontal + 100 sagittal gait points)
    # Hidden layers: progressively smaller
    # Output: 1 neuron for binary classification
    layers_dims = [200, 128, 64, 32, 16, 1]
    
    # Hyperparameters - you can experiment with these
    learning_rate = 0.01    # Start with 0.01, can try 0.005 or 0.02
    num_iterations = 2500   # Can increase if underfitting
    lambd = 0.01           # L2 regularization to prevent overfitting
    
    # Train the model
    parameters, scaler, costs = train_gait_classifier(
        X_train, X_test, y_train, y_test,
        layers_dims=layers_dims,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        lambd=lambd
    )
    
    # ===========================
    # STEP 4: ANALYZE RESULTS
    # ===========================
    
    # Feature importance analysis
    print("\n=== Feature Importance Analysis ===")
    feature_names = df_final.drop(columns=['subject', 'group']).columns.tolist()
    importance_df = analyze_feature_importance(parameters, feature_names)
    
    # Show top 10 most important features
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Analyze frontal vs sagittal plane importance
    frontal_features = [f for f in feature_names if f.startswith('X_')]
    sagittal_features = [f for f in feature_names if f.startswith('Y_')]
    
    frontal_importance = importance_df[importance_df['feature'].isin(frontal_features)]['importance'].mean()
    sagittal_importance = importance_df[importance_df['feature'].isin(sagittal_features)]['importance'].mean()
    
    print(f"\nAverage Frontal Plane Feature Importance: {frontal_importance:.4f}")
    print(f"Average Sagittal Plane Feature Importance: {sagittal_importance:.4f}")
    
    # ===========================
    # STEP 5: MAKE PREDICTIONS ON NEW DATA
    # ===========================
    
    def predict_new_subject(new_data, parameters, scaler):
        """
        Predict knee OA for a new subject
        
        Arguments:
        new_data -- numpy array of shape (200,) with gait features
        parameters -- trained model parameters
        scaler -- fitted StandardScaler from training
        
        Returns:
        prediction -- 0 (healthy) or 1 (OA)
        probability -- probability of having OA
        """
        # Preprocess the new data
        new_data_scaled = scaler.transform(new_data.reshape(1, -1))
        new_data_formatted = new_data_scaled.T
        
        # Make prediction
        prediction, probability = predict(new_data_formatted, parameters)
        
        return prediction[0, 0], probability[0, 0]
    
    # Example usage (uncomment when you have new data):
    # new_subject_data = np.random.randn(200)  # Replace with actual data
    # pred, prob = predict_new_subject(new_subject_data, parameters, scaler)
    # print(f"Prediction: {'OA' if pred == 1 else 'Healthy'}, Probability: {prob:.3f}")
    
    return parameters, scaler, costs, importance_df

# ===========================
# STEP 6: MODEL VALIDATION AND IMPROVEMENT
# ===========================

def hyperparameter_search():
    """
    Perform simple hyperparameter search to find optimal settings
    """
    
    print("\n=== Hyperparameter Search ===")
    
    # Define parameter grid
    learning_rates = [0.005, 0.01, 0.02]
    regularization_params = [0.0, 0.01, 0.05]
    architectures = [
        [200, 128, 64, 1],      # Shallow
        [200, 128, 64, 32, 1],  # Medium
        [200, 128, 64, 32, 16, 1]  # Deep
    ]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    for lr in learning_rates:
        for lambd in regularization_params:
            for arch in architectures:
                print(f"\nTesting: LR={lr}, L2={lambd}, Architecture={arch}")
                
                try:
                    # Train model with current parameters
                    parameters, scaler, costs = train_gait_classifier(
                        X_train, X_test, y_train, y_test,
                        layers_dims=arch,
                        learning_rate=lr,
                        num_iterations=1500,  # Reduced for faster search
                        lambd=lambd
                    )
                    
                    # Evaluate performance
                    X_test_proc, _, y_test_proc, _, _ = preprocess_gait_data(
                        X_train, X_test, y_train, y_test
                    )
                    test_accuracy, test_auc, _, _ = evaluate_model(X_test_proc, y_test_proc, parameters)
                    
                    # Store results
                    result = {
                        'learning_rate': lr,
                        'lambda': lambd,
                        'architecture': arch,
                        'test_accuracy': test_accuracy,
                        'test_auc': test_auc
                    }
                    results.append(result)
                    
                    # Check if this is the best so far
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        best_params = result.copy()
                        
                    print(f"Test Accuracy: {test_accuracy:.2f}%, AUC: {test_auc:.4f}")
                    
                except Exception as e:
                    print(f"Error with parameters: {e}")
                    continue
    
    print(f"\n=== Best Parameters Found ===")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Best Parameters: {best_params}")
    
    # Convert results to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    return results_df, best_params

# ===========================
# STEP 7: CROSS-VALIDATION
# ===========================

def cross_validate_model(n_splits=5):
    """
    Perform stratified k-fold cross-validation
    """
    from sklearn.model_selection import StratifiedKFold
    
    print(f"\n=== {n_splits}-Fold Cross-Validation ===")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_accuracies = []
    cv_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        # Train model
        parameters, scaler, _ = train_gait_classifier(
            X_train_cv, X_val_cv, y_train_cv, y_val_cv,
            layers_dims=[200, 128, 64, 32, 1],
            learning_rate=0.01,
            num_iterations=2000,
            lambd=0.01
        )
        
        # Evaluate
        X_val_proc, _, y_val_proc, _, _ = preprocess_gait_data(
            X_train_cv, X_val_cv, y_train_cv, y_val_cv
        )
        val_accuracy, val_auc, _, _ = evaluate_model(X_val_proc, y_val_proc, parameters)
        
        cv_accuracies.append(val_accuracy)
        cv_aucs.append(val_auc)
        
        print(f"Fold {fold + 1} - Accuracy: {val_accuracy:.2f}%, AUC: {val_auc:.4f}")
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Mean Accuracy: {np.mean(cv_accuracies):.2f}% ± {np.std(cv_accuracies):.2f}%")
    print(f"Mean AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    
    return cv_accuracies, cv_aucs

# ===========================
# STEP 8: ADVANCED ANALYSIS
# ===========================

def analyze_gait_patterns(parameters, scaler, X_train, y_train):
    """
    Analyze what gait patterns the model has learned
    """
    print("\n=== Gait Pattern Analysis ===")
    
    # Get feature names
    feature_names = df_final.drop(columns=['subject', 'group']).columns.tolist()
    
    # Separate frontal and sagittal features
    frontal_features = [f for f in feature_names if f.startswith('X_')]
    sagittal_features = [f for f in feature_names if f.startswith('Y_')]
    
    # Get average patterns for each group
    X_train_scaled = scaler.transform(X_train)
    
    healthy_mask = y_train == 'healthy'  # Adjust based on your labels
    oa_mask = y_train == 'OA'  # Adjust based on your labels
    
    healthy_avg = X_train_scaled[healthy_mask].mean(axis=0)
    oa_avg = X_train_scaled[oa_mask].mean(axis=0)
    
    # Plot average gait patterns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Frontal plane (first 100 features)
    ax1.plot(healthy_avg[:100], label='Healthy', color='blue', alpha=0.7)
    ax1.plot(oa_avg[:100], label='OA', color='red', alpha=0.7)
    ax1.set_title('Average Frontal Plane Gait Pattern')
    ax1.set_xlabel('Gait Cycle (%)')
    ax1.set_ylabel('Normalized Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sagittal plane (next 100 features)
    ax2.plot(healthy_avg[100:], label='Healthy', color='blue', alpha=0.7)
    ax2.plot(oa_avg[100:], label='OA', color='red', alpha=0.7)
    ax2.set_title('Average Sagittal Plane Gait Pattern')
    ax2.set_xlabel('Gait Cycle (%)')
    ax2.set_ylabel('Normalized Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate differences
    difference_frontal = oa_avg[:100] - healthy_avg[:100]
    difference_sagittal = oa_avg[100:] - healthy_avg[100:]
    
    # Plot differences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(difference_frontal, color='purple', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('Difference in Frontal Plane (OA - Healthy)')
    ax1.set_xlabel('Gait Cycle (%)')
    ax1.set_ylabel('Amplitude Difference')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(difference_sagittal, color='orange', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Difference in Sagittal Plane (OA - Healthy)')
    ax2.set_xlabel('Gait Cycle (%)')
    ax2.set_ylabel('Amplitude Difference')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return healthy_avg, oa_avg, difference_frontal, difference_sagittal

def plot_learning_curves_comparison(results_list, labels):
    """
    Compare learning curves from different model configurations
    """
    plt.figure(figsize=(12, 8))
    
    for costs, label in zip(results_list, labels):
        plt.plot(costs, label=label, linewidth=2)
    
    plt.xlabel('Iterations (per hundreds)')
    plt.ylabel('Cost')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def generate_model_report(parameters, scaler, costs, importance_df, 
                         test_accuracy, test_auc, cv_results=None):
    """
    Generate a comprehensive model report
    """
    print("\n" + "="*60)
    print("           GAIT WAVEFORM CLASSIFICATION REPORT")
    print("="*60)
    
    # Model Architecture
    L = len(parameters) // 2
    print(f"\nMODEL ARCHITECTURE:")
    print(f"  - Number of layers: {L}")
    print(f"  - Input features: {parameters['W1'].shape[1]}")
    for i in range(1, L):
        print(f"  - Hidden layer {i}: {parameters[f'W{i}'].shape[0]} neurons")
    print(f"  - Output layer: {parameters[f'W{L}'].shape[0]} neuron(s)")
    
    # Training Performance
    print(f"\nTRAINING PERFORMANCE:")
    print(f"  - Final training cost: {costs[-1]:.6f}")
    print(f"  - Training converged: {'Yes' if costs[-1] < costs[0] * 0.1 else 'No'}")
    
    # Test Performance
    print(f"\nTEST PERFORMANCE:")
    print(f"  - Test Accuracy: {test_accuracy:.2f}%")
    print(f"  - Test AUC: {test_auc:.4f}")
    
    # Cross-validation (if available)
    if cv_results:
        cv_acc, cv_auc = cv_results
        print(f"\nCROSS-VALIDATION RESULTS:")
        print(f"  - Mean CV Accuracy: {np.mean(cv_acc):.2f}% ± {np.std(cv_acc):.2f}%")
        print(f"  - Mean CV AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    
    # Feature Importance
    print(f"\nFEATURE IMPORTANCE:")
    frontal_features = [f for f in importance_df['feature'] if f.startswith('X_')]
    sagittal_features = [f for f in importance_df['feature'] if f.startswith('Y_')]
    
    frontal_importance = importance_df[importance_df['feature'].isin(frontal_features)]['importance'].mean()
    sagittal_importance = importance_df[importance_df['feature'].isin(sagittal_features)]['importance'].mean()
    
    print(f"  - Average frontal plane importance: {frontal_importance:.4f}")
    print(f"  - Average sagittal plane importance: {sagittal_importance:.4f}")
    print(f"  - Most important plane: {'Frontal' if frontal_importance > sagittal_importance else 'Sagittal'}")
    
    print(f"\nTOP 5 MOST IMPORTANT FEATURES:")
    for i, row in importance_df.head(5).iterrows():
        plane = "Frontal" if row['feature'].startswith('X_') else "Sagittal"
        time_point = row['feature'].split('_')[1]
        print(f"  {i+1}. {row['feature']} ({plane} plane, {time_point}% gait cycle) - {row['importance']:.4f}")
    
    print("\n" + "="*60)

# ===========================
# STEP 9: MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    """
    Main execution script - run this to perform complete analysis
    """
    
    print("Starting Gait Waveform Classification Analysis...")
    
    # Step 1: Basic model training
    print("\n### STEP 1: Basic Model Training ###")
    parameters, scaler, costs, importance_df = run_gait_classification_pipeline()
    
    # Step 2: Cross-validation
    print("\n### STEP 2: Cross-Validation ###")
    cv_accuracies, cv_aucs = cross_validate_model(n_splits=5)
    
    # Step 3: Hyperparameter search (optional - takes longer)
    # print("\n### STEP 3: Hyperparameter Search ###")
    # results_df, best_params = hyperparameter_search()
    
    # Step 4: Gait pattern analysis
    print("\n### STEP 4: Gait Pattern Analysis ###")
    healthy_avg, oa_avg, diff_frontal, diff_sagittal = analyze_gait_patterns(
        parameters, scaler, X_train, y_train
    )
    
    # Step 5: Generate comprehensive report
    print("\n### STEP 5: Final Report ###")
    
    # Get test performance
    X_train_proc, X_test_proc, y_train_proc, y_test_proc, _ = preprocess_gait_data(
        X_train, X_test, y_train, y_test
    )
    test_accuracy, test_auc, _, _ = evaluate_model(X_test_proc, y_test_proc, parameters)
    
    generate_model_report(
        parameters, scaler, costs, importance_df, 
        test_accuracy, test_auc, 
        cv_results=(cv_accuracies, cv_aucs)
    )
    
    print("\nAnalysis complete! Check the plots and reports above.")
    print("Model parameters and scaler have been saved for future predictions.")

# ===========================
# STEP 10: DEPLOYMENT HELPER
# ===========================

def save_model_for_deployment(parameters, scaler, filename_prefix="gait_model"):
    """
    Save trained model for later use
    """
    import pickle
    
    # Save parameters
    with open(f"{filename_prefix}_parameters.pkl", "wb") as f:
        pickle.dump(parameters, f)
    
    # Save scaler
    with open(f"{filename_prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Model saved as {filename_prefix}_parameters.pkl and {filename_prefix}_scaler.pkl")

def load_model_for_prediction(filename_prefix="gait_model"):
    """
    Load trained model for predictions
    """
    import pickle
    
    # Load parameters
    with open(f"{filename_prefix}_parameters.pkl", "rb") as f:
        parameters = pickle.load(f)
    
    # Load scaler
    with open(f"{filename_prefix}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return parameters, scaler

# Example usage for saving/loading:
# save_model_for_deployment(parameters, scaler)
# loaded_params, loaded_scaler = load_model_for_prediction()