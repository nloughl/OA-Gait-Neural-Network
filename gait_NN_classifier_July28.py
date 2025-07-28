# Gait Waveform Neural Network Classifier
# Adapted from deep learning image classification for time-series gait data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns


# ACTIVATION FUNCTIONS

def sigmoid(Z):
    """Sigmoid activation function"""
    A = 1/(1+np.exp(-np.clip(Z, -500, 500))) 
    cache = Z
    return A, cache

def relu(Z):
    """ReLU activation function"""
    A = np.maximum(0, Z)
    cache = Z 
    return A, cache

def sigmoid_backward(dA, cache):
    """Backward propagation for sigmoid"""
    Z = cache
    s = 1/(1+np.exp(-np.clip(Z, -500, 500)))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, cache):
    """Backward propagation for ReLU"""
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

# PARAMETER INITIALIZATION

def initialize_parameters_deep(layer_dims):
    """
    Initialize parameters for L-layer neural network with Xavier initialization
    
    Arguments:
    layer_dims -- list containing dimensions of each layer
    
    Returns:
    parameters -- dictionary containing parameters W1, b1, ..., WL, bL
    """
    np.random.seed(42)  # For reproducibility
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        # Xavier initialization for better convergence
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


# FORWARD PROPAGATION

def linear_forward(A, W, b):
    """Linear part of forward propagation"""
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """Linear->Activation forward propagation"""
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    """
    Forward propagation for [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    
    Arguments:
    X -- input data of shape (input_size, number_of_examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches for backward propagation
    """
    caches = []
    A = X
    L = len(parameters) // 2
    
    # [LINEAR -> RELU] * (L-1)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, 
                                           parameters['W' + str(l)], 
                                           parameters['b' + str(l)], 
                                           activation="relu")
        caches.append(cache)
    
    # LINEAR -> SIGMOID
    AL, cache = linear_activation_forward(A, 
                                        parameters['W' + str(L)], 
                                        parameters['b' + str(L)], 
                                        activation="sigmoid")
    caches.append(cache)
    
    return AL, caches


# COST FUNCTION

def compute_cost(AL, Y, parameters=None, lambd=0.0):
    """
    Compute cost with optional L2 regularization
    
    Arguments:
    AL -- probability vector corresponding to label predictions
    Y -- true label vector
    parameters -- dictionary of parameters (for regularization)
    lambd -- regularization parameter
    
    Returns:
    cost -- cross-entropy cost with regularization
    """
    m = Y.shape[1]
    
    # Cross-entropy cost
    logprobs = np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL))
    cost = -np.sum(logprobs) / m
    
    # L2 regularization
    if lambd > 0 and parameters is not None:
        L = len(parameters) // 2
        L2_regularization_cost = 0
        for l in range(1, L+1):
            L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
        L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost
        cost = cost + L2_regularization_cost
    
    cost = np.squeeze(cost)
    return cost


# BACKWARD PROPAGATION

def linear_backward(dZ, cache):
    """Linear portion of backward propagation"""
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """Linear->Activation backward propagation"""
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, parameters=None, lambd=0.0):
    """
    Backward propagation for [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID
    with optional L2 regularization
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Initialize backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    
    # Lth layer (SIGMOID -> LINEAR)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
    # Add L2 regularization to dW
    if lambd > 0 and parameters is not None:
        grads["dW" + str(L)] += (lambd / m) * parameters['W' + str(L)]
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
        # Add L2 regularization to dW
        if lambd > 0 and parameters is not None:
            grads["dW" + str(l + 1)] += (lambd / m) * parameters['W' + str(l + 1)]

    return grads


# PARAMETER UPDATE

def update_parameters(parameters, grads, learning_rate):
    """Update parameters using gradient descent"""
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


# PREDICTION AND EVALUATION

def predict(X, parameters):
    """Make predictions using trained parameters"""
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5).astype(int)
    return predictions, AL

def evaluate_model(X, y, parameters):
    """Evaluate model performance"""
    predictions, probabilities = predict(X, parameters)
    accuracy = np.mean(predictions == y) * 100
    auc = roc_auc_score(y.T, probabilities.T)
    return accuracy, auc, predictions, probabilities


# MAIN TRAINING FUNCTION

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, 
                  print_cost=True, lambd=0.0):
    """
    Implements a L-layer neural network
    
    Arguments:
    X -- data, numpy array of shape (number of features, number of examples)
    Y -- true "label" vector, of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    lambd -- regularization parameter
    
    Returns:
    parameters -- parameters learnt by the model
    costs -- list of costs during training
    """
    
    np.random.seed(42)
    costs = []
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y, parameters, lambd)
        
        # Backward propagation
        grads = L_model_backward(AL, Y, caches, parameters, lambd)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    return parameters, costs


# DATA PREPROCESSING FOR GAIT

def preprocess_gait_data(X_train, X_test, y_train, y_test):
    """
    Preprocess gait waveform data for neural network
    
    Arguments:
    X_train, X_test -- feature matrices
    y_train, y_test -- label vectors
    
    Returns:
    Preprocessed and properly shaped data
    """
    
    # Standardize features (important for gait waveforms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for neural network (features, examples)
    X_train_final = X_train_scaled.T
    X_test_final = X_test_scaled.T
    
    # Convert labels to binary ('healthy' = 0, 'OA' = 1)
    label_map = {'healthy': 0, 'OA': 1}  # Modify as needed
    if isinstance(y_train[0], str):
        y_train_binary = np.array([label_map[label] for label in y_train])
        y_test_binary = np.array([label_map[label] for label in y_test])
    else:
        y_train_binary = y_train
        y_test_binary = y_test
    
    # Reshape labels (1, examples)
    y_train_final = y_train_binary.reshape(1, -1)
    y_test_final = y_test_binary.reshape(1, -1)
    
    return X_train_final, X_test_final, y_train_final, y_test_final, scaler


# VISUALIZATION FUNCTIONS

def plot_cost(costs):
    """Plot the cost function over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title('Learning curve')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['Healthy', 'OA']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true.T, y_pred.T)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def analyze_feature_importance(parameters, feature_names=None):
    """
    Simple feature importance analysis based on first layer weights
    """
    W1 = parameters['W1']
    # Average absolute weights across all neurons in first hidden layer
    importance = np.mean(np.abs(W1), axis=0)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importance))]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Average Absolute Weight')
    plt.title('Feature Importance (Top 20)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df


# EXAMPLE USAGE

def train_gait_classifier(X_train, X_test, y_train, y_test, 
                         layers_dims=None, learning_rate=0.0075, 
                         num_iterations=2000, lambd=0.01):
    """
    Complete pipeline for training gait waveform classifier
    """
    
    print("=== Gait Waveform Classification with Deep Neural Network ===\n")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_proc, X_test_proc, y_train_proc, y_test_proc, scaler = \
        preprocess_gait_data(X_train, X_test, y_train, y_test)
    
    print(f"Training set: {X_train_proc.shape[1]} examples")
    print(f"Test set: {X_test_proc.shape[1]} examples")
    print(f"Features: {X_train_proc.shape[0]}\n")
    
    # Set default architecture if not provided
    if layers_dims is None:
        layers_dims = [X_train_proc.shape[0], 128, 64, 32, 1]  # Input -> Hidden layers -> Output
    
    print(f"Network architecture: {layers_dims}")
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {num_iterations}")
    print(f"L2 regularization parameter: {lambd}\n")
    
    # Train the model
    print("Training model...")
    parameters, costs = L_layer_model(X_train_proc, y_train_proc, layers_dims, 
                                    learning_rate=learning_rate, 
                                    num_iterations=num_iterations, 
                                    print_cost=True, lambd=lambd)
    
    # Plot learning curve
    plot_cost(costs)
    
    # Evaluate on training set
    print("\n=== Training Set Performance ===")
    train_accuracy, train_auc, train_pred, train_prob = evaluate_model(X_train_proc, y_train_proc, parameters)
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Training AUC: {train_auc:.4f}")
    
    # Evaluate on test set
    print("\n=== Test Set Performance ===")
    test_accuracy, test_auc, test_pred, test_prob = evaluate_model(X_test_proc, y_test_proc, parameters)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Classification report
    print("\n=== Detailed Classification Report ===")
    print(classification_report(y_test_proc.T, test_pred.T, target_names=['Healthy', 'OA']))
    
    # Confusion matrix
    plot_confusion_matrix(y_test_proc, test_pred)
    
    return parameters, scaler, costs

# Example of how to use with the gait data  
"""
# After your data preparation code:
parameters, scaler, costs = train_gait_classifier(
    X_train, X_test, y_train, y_test,
    layers_dims=[200, 128, 64, 32, 1],  # 200 input features
    learning_rate=0.01,
    num_iterations=2000,
    lambd=0.01
)

# Feature importance analysis
feature_names = df_final.drop(columns=['subject', 'group']).columns.tolist()
importance_df = analyze_feature_importance(parameters, feature_names)
"""