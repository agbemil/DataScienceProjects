# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:59:23 2023

@author: agbem
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample



# Read the data
df = pd.read_csv(r"D:/Third Year/Asmptotic/train_CSV.csv")
test = pd.read_csv(r"D:/Third Year/Asmptotic/test_csv.csv")

###### Data Preprocessing ############

# Separate the majority and minority classes
class_counts = df['label'].value_counts()
min_class = class_counts.idxmin()
min_class_count = class_counts.min()

# Downsampling
df_downsampled = pd.DataFrame()  # Initialize an empty dataframe to hold the downsampled data

for class_label in df['label'].unique():
    class_subset = df[df['label'] == class_label]
    df_downsampled = pd.concat([df_downsampled, resample(class_subset,
                                                         replace=False,  # sample without replacement
                                                         n_samples=min_class_count,  # to match minority class
                                                         random_state=123)])  # reproducible results

# Shuffle the dataset
df_downsampled = df_downsampled.sample(frac=1, random_state=123).reset_index(drop=True)

# Continue with preprocessing
y_train = df_downsampled['label']
X_train = df_downsampled.drop(columns=['label'])
y_test = test['label']
X_test = test.drop(columns=['label'])


# One-hot encoding
y_train_encoded = pd.get_dummies(y_train).values
y_test_encoded = pd.get_dummies(y_test).values

# Standardization
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]



# Loss Function (Softmax)
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

# Gradient function 
def gradient(X, y, w):
    m = len(y)
    h = softmax(X @ w)
    return -1/m * X.T @ (y - h)

# Soft Threshold Function
def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

# Computing the cost
def cost_function(X, y, w):
    h = softmax(X @ w)
    return -np.mean(np.sum(y * np.log(h), axis=1))

# Forward Backward Splitting  Algorithm
def forward_backward_splitting(X, y, lam, alpha, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    n_classes = y.shape[1]
    w = np.zeros((n_features, n_classes))
    costs = []
    
    for i in range(max_iter):
        w_temp = w - alpha * gradient(X, y, w)
        w_new = soft_thresholding(w_temp, alpha * lam)

        # Calculate and record the cost
        cost = cost_function(X, y, w_new)
        costs.append(cost)

        # Check for convergence based on cost change
        if i > 0 and abs(costs[i] - costs[i-1]) < tol:
            print(f"Converged at iteration {i}")
            break

        # Update weights for next iteration
        w = w_new
        
    # If convergence criterion was never met, output the total iterations
    if i == max_iter - 1:
        print(f"Reached maximum iterations: {i+1}")
    
    return w, costs

# Hyperparameter tuning using K-Fold cross-validation
# =============================================================================
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# lams = [0.01, 0.1, 1]
# alphas = [0.001, 0.01, 0.1]
# 
# best_acc = 0
# best_params = {}
# 
# for lam in lams:
#     for alpha in alphas:
#         fold_accs = []
#         for train_idx, val_idx in kfold.split(X_train, y_train):
#             X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
#             y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
#             
#             w, _ = forward_backward_splitting(X_train_fold, y_train_fold, lam, alpha)
#             predictions = np.argmax(softmax(X_val_fold @ w), axis=1)
#             true_labels = np.argmax(y_val_fold, axis=1)
#             acc = accuracy_score(true_labels, predictions)
#             fold_accs.append(acc)
#         
#         mean_acc = np.mean(fold_accs)
#         if mean_acc > best_acc:
#             best_acc = mean_acc
#             best_params = {"lam": lam, "alpha": alpha}
# 
# # Print the best parameters
# print(f"Best parameters are: Lambda = {best_params['lam']}, Alpha = {best_params['alpha']}")
# 
# =============================================================================

 
Lambda = 0.01
Alpha = 0.1
# Training with the best parameters

w, costs = forward_backward_splitting(X_train, y_train_encoded, Lambda, Alpha)

# Convergence plot
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Convergence of Forward-Backward Splitting")
plt.show()

# Predictions
probas = softmax(X_test @ w)
predictions = np.argmax(probas, axis=1)
true_labels = np.argmax(y_test_encoded, axis=1)
print("Accuracy:", accuracy_score(true_labels, predictions))

# AUC plot
plt.figure(figsize=(10, 8))
for i in range(y_train_encoded.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_encoded[:, i], probas[:, i])
    auc_score = roc_auc_score(y_test_encoded[:, i], probas[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}, AUC: {auc_score:.2f}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (FBS)')
plt.legend()
plt.show()


# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)

# Specificity and Sensitivity
sensitivity = np.diag(cm) / np.sum(cm, axis=1)
specificity = np.diag(cm) / np.sum(cm, axis=0)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=np.unique(y_test)).plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()



################Subgradient Descesnt #################
# Define the softmax function

def softmas(scores):
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Updated logistic loss function with L1 regularization
def logistic_loss(X, y_encoded, weights, lambda_reg):
    scores = np.dot(X, weights)
    probs = softmas(scores)
    core_loss = -np.mean(np.sum(y_encoded * np.log(probs + 1e-8), axis=1))  # Adding a small value for numerical stability
    reg_loss = lambda_reg * np.sum(np.abs(weights))
    return core_loss + reg_loss

# Updated subgradient computation
def compute_subgradient(X, y_encoded, weights, lambda_reg):
    scores = np.dot(X, weights)
    probs = softmas(scores)
    gradient = np.dot(X.T, (probs - y_encoded)) / len(y_encoded)
    subgrad = gradient + lambda_reg * np.sign(weights)
    return subgrad

# Subgradient descent
def subgradient_descent(X, y_encoded, lambda_reg, learning_rate=0.1, max_iter=1000, tol=1e-4):
    weights = np.zeros((X.shape[1], y_encoded.shape[1]))
    loss_history = []
    for i in range(max_iter):
        subgrad = compute_subgradient(X, y_encoded, weights, lambda_reg)
        weights -= learning_rate * subgrad
        loss = logistic_loss(X, y_encoded, weights, lambda_reg)
        loss_history.append(loss)
        
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"Converged at iteration {i}")
            break
    
    return weights, loss_history


# Train the model
lambda_reg = 0.01
num_classes = y_train_encoded.shape[1]
weights, loss_history = subgradient_descent(X_train, y_train_encoded, lambda_reg)

# Plot convergence curve
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Convergence Curve')
plt.show()


# Prediction and Performance Metrics
def predict(X, weights):
    scores = np.dot(X, weights)
    probs = softmas(scores)
    return np.argmax(probs, axis=1)

y_pred = predict(X_test, weights)
accuracy = accuracy_score(np.argmax(y_test_encoded, axis=1), y_pred)
print("Accuracy:", accuracy)

# Compute confusion matrix for sensitivity and specificity
cm = confusion_matrix(np.argmax(y_test_encoded, axis=1), y_pred)
sensitivity = np.diag(cm) / np.sum(cm, axis=1, where=np.sum(cm, axis=1) != 0)
specificity = np.diag(cm) / np.sum(cm, axis=0, where=np.sum(cm, axis=0) != 0)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# ROC and AUC for multi-class
y_test_binarized = label_binarize(np.argmax(y_test_encoded, axis=1), classes=np.unique(np.argmax(y_train_encoded, axis=1)))
y_pred_proba = softmas(np.dot(X_test, weights))

for i in range(num_classes):
    if np.sum(y_test_binarized[:, i]) > 0:
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (area = {auc:.2f})')
    else:
        print(f"Class {i} not present in the test set")
        
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (Sub-gradient Descent)')
plt.legend(loc='best')
plt.show()


################ Combining plots side by side ####################

#Convergence 
plt.figure(figsize=(10, 6))

# Plot for Forward-Backward Splitting
plt.plot(costs, label='Forward-Backward Splitting')

# Plot for the other method
plt.plot(loss_history, label='Subgradient Descent')

plt.xlabel('Iteration')
plt.ylabel('Loss/Cost')
plt.title('Comparison of Convergence Curves')
plt.legend()
plt.show()

 
# Accuracy 
accuracy_fbs = 0.9270444519850696
accuracy_subgradient = 0.8907363420427553

# Method names
methods = ['FBS', 'Subgradient']

# Plot
plt.figure(figsize=(8, 4))
plt.bar(methods, [accuracy_fbs, accuracy_subgradient], color=['blue', 'green'])

# Adding details
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies')
plt.ylim(0.85, 0.95)  # Set y-axis limits for better comparison
for i, v in enumerate([accuracy_fbs, accuracy_subgradient]):
    plt.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom')

plt.show()


# ROC plot for FBS & Sub-gradient Descent
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_encoded[:, i], probas[:, i])
    auc_score = roc_auc_score(y_test_encoded[:, i], probas[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}, AUC: {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (FBS)')
plt.legend()
 
plt.subplot(1, 2, 2)
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC: {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (Sub-gradient Descent)')
plt.legend()
plt.show()


# Calculate total errors for FBS
total_errors_fbs = np.sum(predictions != true_labels)
true_labels_subgradient = np.argmax(y_test_encoded, axis=1)
total_errors_subgradient = np.sum(y_pred != true_labels_subgradient)
print("Total Errors (FBS):", total_errors_fbs)
print("Total Errors (Sub-gradient Descent):", total_errors_subgradient)


# Total number of samples in the test set
total_samples = len(true_labels)

# Calculate total errors for FBS
total_errors_fbs = np.sum(predictions != true_labels)
error_percentage_fbs = (total_errors_fbs / total_samples) * 100

# Calculate total errors for Sub-gradient Descent
total_errors_subgradient = np.sum(y_pred != true_labels_subgradient)
error_percentage_subgradient = (total_errors_subgradient / total_samples) * 100
print("Total Error Percentage (FBS):", error_percentage_fbs, "%")
print("Total Error Percentage (Sub-gradient Descent):", error_percentage_subgradient, "%")
