# Standard Library Imports
import copy  # For making deep copies of mutable objects
import warnings  # For suppressing warnings

# Data Manipulation and Visualization Libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Data visualization
from tabulate import tabulate  # For printing data in table format

# Scikit-learn: Model Selection, Metrics, and Preprocessing
from sklearn.model_selection import GridSearchCV  # Hyperparameter tuning
from sklearn.dummy import DummyClassifier, DummyRegressor  # Baseline models
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc,  # Classification metrics
                             mean_squared_error, r2_score)  # Regression metrics
from sklearn.preprocessing import PolynomialFeatures  # For generating polynomial features

# Imbalanced-learn: Pipelines and Resampling Techniques
from imblearn.pipeline import Pipeline  # Pipeline to include resampling steps
from imblearn.under_sampling import RandomUnderSampler  # For undersampling imbalanced datasets
from imblearn.over_sampling import SMOTE  # For oversampling imbalanced datasets

# Scikit-learn: Model Inspection
from sklearn.inspection import permutation_importance  # Permutation importance for model interpretation

class ModelWrapper:
    def __init__(self, estimator, param_grid, name):
        """
        Class that encapsulates a model, its hyperparameters for grid search, and its name.

        Args:
            model (estimator): The model (e.g., sklearn estimator) to be tuned.
            param_grid (dict): The hyperparameters to use in GridSearchCV.
            name (str): The name of the model.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.name = name

def plot_feature_importance(model, X_train, X_test, y_test, n_repeats=10, random_state=42):
    """
    Function to calculate and plot the feature importance for a given model.
    
    Parameters:
    - model: The trained model to evaluate.
    - X_train: Training data used to extract feature names.
    - X_test: Test data to calculate permutation importance.
    - y_test: True labels for the test data.
    - n_repeats: Number of times to shuffle a feature for importance calculation (default 10).
    - random_state: Random seed for reproducibility (default 42).
    """
    estimator = model['Model']
    # Calculate permutation importance
    result = permutation_importance(estimator, X_test, y_test, n_repeats=n_repeats, random_state=random_state)
    
    # Create a dataframe to display feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,  # Assumes X_train is a DataFrame
        'Importance': result.importances_mean
    })
    
    # Sort by importance and display the result
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(sorted_df)
    
    model_name = model['Model Name']
    # Optional: Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df['Feature'], sorted_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance for {model_name} Model')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most important at the top
    plt.show()

def evaluate_models(best_models, X, y, model_type='classification', scoring=None, baseline_model=None):
    """
    Evaluates the best models on the given feature set (X) and target values (y),
    and displays a table with model performance. Always displays the best model 
    per type based on the chosen scoring metric. If scoring is provided, models are 
    ordered by the selected metric, and the top model is marked with an asterisk (*).

    Args:
        best_models (list): List of dictionaries containing best models and associated metadata.
        X (pd.DataFrame or np.array): Feature set (can be training or test).
        y (pd.Series or np.array): Target values (can be training or test).
        model_type (str): The type of model ('classification' or 'regression').
        scoring (str): The metric to score models on ('accuracy', 'f1', 'precision', 'recall' for classification,
                      'rmse', 'r2', 'mae' for regression).

    Returns:
        None
    """
    # Check if the model type is valid
    if model_type not in ['classification', 'regression']:
        raise ValueError("model_type must be 'classification' or 'regression'")
    
    # Insert the baseline model at the beginning of the list
    if baseline_model is not None:
        best_models.insert(0, baseline_model)

    # Initialize the results table that will store the results of each model
    results_table = []

    # Evaluate each model
    for model_info in best_models:
        model = model_info['Model']
        model_name = model_info['Model Name']
        best_params = model_info['Best Parameters']

        # Predictions on the feature set
        y_pred = model.predict(X)

        if model_type == 'classification':
            # Calculate classification metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

            # Add the model's performance metrics to the results table
            results_table.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Best Parameters': best_params
            })
        else:
            # Calculate regression metrics
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            mae = np.mean(np.abs(y - y_pred))

            # Add the model's performance metrics to the results table 
            results_table.append({
                'Model': model_name,
                'RMSE': rmse,
                'R²': r2,
                'MAE': mae,
                'Best Parameters': best_params
            })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results_table)

    # Determine sorting by problem type and then sort the results dataframe by the scoring metric
    if scoring:
        if model_type == 'classification':
            if scoring == 'accuracy':
                results_df = results_df.sort_values(by='Accuracy', ascending=False)
            elif scoring == 'precision':
                results_df = results_df.sort_values(by='Precision', ascending=False)
            elif scoring == 'recall':
                results_df = results_df.sort_values(by='Recall', ascending=False)
            elif scoring == 'f1':
                results_df = results_df.sort_values(by='F1 Score', ascending=False)
            else:
                raise ValueError(f"Invalid scoring parameter for classification: {scoring}")
        else:
            if scoring == 'rmse':
                results_df = results_df.sort_values(by='RMSE', ascending=True)  # For RMSE, lower is better
            elif scoring == 'r2':
                results_df = results_df.sort_values(by='R²', ascending=False)
            elif scoring == 'mae':
                results_df = results_df.sort_values(by='MAE', ascending=True)  # For MAE, lower is better
            else:
                raise ValueError(f"Invalid scoring parameter for regression: {scoring}")
    else:
        # Default to sorting by the first metric in the table
        results_df = results_df.sort_values(by=results_df.columns[1], ascending=False)

    # Add an asterisk to the top model's name
    results_df.iloc[0, results_df.columns.get_loc('Model')] += ' *'

    # Display the ordered model results with the top one marked
    print("\nBest Model Evaluation Results (Top model marked with *):")
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False, numalign='right', floatfmt=".4f"))

    # Generate appropriate charts based on the model type
    if model_type == 'classification':
        # Bar chart for classification models
        results_df.set_index('Model', inplace=True)
        results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', figsize=(12, 6))
        plt.title('Classification Model Performance')
        plt.xlabel('Model')
        plt.ylabel('Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        # Bar chart for regression models
        results_df.set_index('Model', inplace=True)
        results_df[['RMSE', 'R²', 'MAE']].plot(kind='bar', figsize=(12, 6), color=['lightcoral', 'lightblue', 'lightgreen'])
        plt.title('Regression Model Performance')
        plt.xlabel('Model')
        plt.ylabel('Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Display the hyperparameters of the models in a readable format
    print("\nModel Hyperparameters:")

    # Loop through each model's information and display its hyperparameters
    for model_index, model_info in enumerate(best_models, start=1):
        # Get the model name
        model_name = model_info['Model Name']
        
        # Get the dictionary of best hyperparameters for this model
        hyperparameters = model_info['Best Parameters']
        
        # Format the hyperparameters as a comma-separated string
        hyperparameters_str = ', '.join([f'{param_name}={param_value}' for param_name, param_value in hyperparameters.items()])
        
        # Print the model's index, name, and its hyperparameters
        print(f"{model_index}. {model_name}: {hyperparameters_str}")



def assign_score_label(scoring):
    """
    Assigns a label to the score based on the scoring method provided.

    Args:
        scoring (str): The scoring method used to evaluate the model.

    Returns:
        str: The label for the score.
    """
    # Assign score label based on the scoring method provided
    if scoring == 'neg_mean_squared_error':
        score_label = 'Train RMSE (CV)'
        best_score = np.sqrt(-best_score)  # Convert neg MSE to RMSE
    elif scoring == 'r2':
        score_label = 'Train R² (CV)'
    elif scoring in ['accuracy', 'f1', 'precision', 'recall']:
        score_label = f'Train {scoring.capitalize()} (CV)'
    else:
        raise ValueError(f"Invalid scoring parameter: {scoring}")
    return score_label

def train_models(model_list: list[ModelWrapper], pipeline, X_train=None, y_train=None, cv=5, scoring=None, verbose=False):
    """
    Trains the models using GridSearchCV for each model in the provided list, 
    and returns a list of dictionaries with the best models and associated metadata.

    Args:
        model_list (list): A list of ModelWithParams instances, each containing a model, hyperparameters, and name.
        X_train (pd.DataFrame or np.array): Training feature set.
        y_train (pd.Series or np.array): Training target values.
        cv (int or cross-validation generator): Number of folds or cross-validation strategy.
        scoring (str or callable): Scoring strategy (e.g., 'accuracy' for classification, 'neg_mean_squared_error' for regression).

    Returns:
        best_models (list): List of dictionaries containing the best models and associated metadata for each model.
    """

    if not verbose:
        warnings.filterwarnings("ignore")

    # Check if training data is provided
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must be provided.")
    
    best_models = []

    # Loop over each model in the model list
    for model_info in model_list:
        estimator = model_info.estimator
        param_grid = model_info.param_grid
        model_name = model_info.name
        print(f"Training {model_name} model...")

        pipelineCopy = copy.deepcopy(pipeline)
        pipelineCopy.set_params(model=estimator)

        # Perform GridSearchCV for the current model
        grid_search = GridSearchCV(pipelineCopy, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Extract the best model, parameters, and score
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        #Assign score label
        score_label = assign_score_label(scoring)

        # Append the best model and its metadata to the result list
        best_models.append({
            'Model Name': model_name,
            'Best Parameters': best_params,
            'Model': best_model,
            score_label: best_score,
            'Score': best_score
        })

    return best_models


def train_baseline_model(X_train=None, y_train=None, cv=5, scoring=None, problem_type='classification', verbose=False):
    """
    Train a baseline model based on the problem type (classification or regression), 
    and perform cross-validation using GridSearchCV to evaluate its performance.

    Args:
        X_train (pd.DataFrame or np.array): The training feature set. Must be provided for training.
        y_train (pd.Series or np.array): The training target values. Must be provided for training.
        cv (int or cross-validation generator): Number of folds or cross-validation strategy. Default is 5.
        scoring (str or callable): Scoring strategy used for cross-validation evaluation.
                                   For classification, common scores include 'accuracy' or 'f1'.
                                   For regression, common scores include 'neg_mean_squared_error' or 'r2'.
        problem_type (str): Defines whether the problem is 'classification' or 'regression'.
                            Default is 'classification'.

    Returns:
        dict: A dictionary containing the following keys:
            - 'Model Name': A string representing the name of the baseline model (e.g., 'Baseline(Classification)' or 'Baseline(Regression)').
            - 'Best Parameters': An empty dictionary since no hyperparameters are tuned in baseline models.
            - 'Model': The trained baseline model object.
            - Score Label: The score achieved during cross-validation (label depends on the scoring metric provided).
    
    Raises:
        ValueError: If the problem type is not 'classification' or 'regression'.
    """

    if not verbose:
        warnings.filterwarnings("ignore")

    if problem_type == 'classification':
        # Create a DummyClassifier that predicts the most frequent class in the dataset
        baseline_model = DummyClassifier(strategy='most_frequent')
        print("Baseline model for classification (predicts the most frequent class) is trained.")
        
    elif problem_type == 'regression':
        # Create a DummyRegressor that predicts the mean value of the target variable
        baseline_model = DummyRegressor(strategy='mean')
        print("Baseline model for regression (predicts the mean value) is trained.")
        
    else:
        # Raise an error if the problem type is not valid
        raise ValueError("Invalid problem type. Choose either 'classification' or 'regression'.")
    
    # Create a pipeline with the baseline model
    pipeline = Pipeline(steps=[('model', baseline_model)])

    # Use the train_modelsx function to train and evaluate the baseline model using cross-validation
    models = train_models(
        [ModelWrapper(baseline_model, {}, f'Baseline({problem_type})')],  # List containing the baseline model
        pipeline,  # Pipeline with the baseline model
        X_train, y_train,  # Training data
        cv=cv,  # Cross-validation strategy
        scoring=scoring  # Scoring method
    )

    # Return the best model and its metadata (first model in the list)
    return models[0]


def plot_roc_curve(models_list, X_test, y_test):
    """
    Function to plot ROC curves for multiple models provided in a list of dictionaries.

    :param models_list: List of dictionaries with each containing:
                        - 'Model Name': Name of the model
                        - 'Best Parameters': Best hyperparameters for the model
                        - 'Model': Trained model object
                        
    :param X_test: Test feature set
    :param y_test: True labels for the test set
    """
    plt.figure(figsize=(10, 8))

    for model_info in models_list:
        model_name = model_info['Model Name']
        model = model_info['Model']

        # Predict probabilities
        y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for this model
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")

    # Plot diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison for All Models')
    plt.legend(loc="lower right")
    plt.show()


def get_best_model(models_list, scoring='accuracy'):
    """
    Function to find and return the model with the best score based on the provided scoring metric.
    
    :param models_list: List of dictionaries where each dictionary contains:
                        - 'Model Name': Name of the model
                        - 'Best Parameters': Best hyperparameters for the model
                        - 'Model': Trained model object
                        - 'Score': Precomputed score of the model (cross-validated or test score)
    :param scoring: The scoring metric ('neg_mean_squared_error', 'r2', 'accuracy', 'f1', 'precision', 'recall').
                    Determines how the score should be interpreted (maximize or minimize).
    :return: Dictionary of the best model and its score based on the specified scoring metric.
    """
    best_model = None
    best_score = None

    # Define if we are maximizing or minimizing based on the scoring metric
    if scoring == 'neg_mean_squared_error':
        # We want to minimize RMSE, so start with the highest possible score
        best_score = float('inf')
    elif scoring in ['r2', 'accuracy', 'f1', 'precision', 'recall']:
        # For these metrics, we want to maximize the score, so start with the lowest possible score
        best_score = float('-inf')
    else:
        raise ValueError("Invalid scoring metric provided!")

    for model in models_list:
        print(model)
        model_score = model['Score']

        # Compare scores based on whether we are maximizing or minimizing
        if (scoring == 'neg_mean_squared_error' and model_score < best_score) or \
           (scoring in ['r2', 'accuracy', 'f1', 'precision', 'recall'] and model_score > best_score):
            best_score = model_score
            best_model = model

    return best_model