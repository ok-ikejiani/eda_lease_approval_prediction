import copy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import numpy as np
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
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

def train_modelsx(model_list: list[ModelWrapper], pipeline, X_train=None, y_train=None, cv=5, scoring=None):
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
    # Check if training data is provided
    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must be provided.")
    
    best_models = []
    print(model_list)
    # Loop over each model in the model list
    for model_info in model_list:
        estimator = model_info.estimator
        param_grid = model_info.param_grid
        model_name = model_info.name
        print(model_name)

        # pipeline = Pipeline(steps=[
        #     ('preprocessor', preprocessor), 
        #     # ('under_sampler', RandomUnderSampler()),
        #     ('smote', SMOTE()),
        #     ('poly', PolynomialFeatures(degree=3)),
        #     ('model', estimator)
        # ])
        # pipeline.steps['model'] = estimator
        # pipelineSteps.append(('model', estimator))
        # print(pipelineSteps)
        # pipeline = Pipeline(pipelineSteps)
        # pipeline.set_params(model=estimator)
        pipelineCopy = copy.deepcopy(pipeline)
        pipelineCopy.set_params(model=estimator)
        print(pipelineCopy)

        # Perform GridSearchCV for the current model
        grid_search = GridSearchCV(pipelineCopy, param_grid, cv=cv, scoring=scoring, n_jobs=-1, error_score='raise')
        grid_search.fit(X_train, y_train)

        # Extract the best model, parameters, and score
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

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

        # Append the best model and its metadata to the result list
        best_models.append({
            'Model Name': model_name,
            'Best Parameters': best_params,
            'Best Pipeline': best_model,
            score_label: best_score
        })

    return best_models


def plot_feature_importance(best_model, X_train, X_test, y_test, n_repeats=10, random_state=42):
    """
    Function to calculate and plot the feature importance for a given model.
    
    Parameters:
    - best_model: The trained model to evaluate.
    - X_train: Training data used to extract feature names.
    - X_test: Test data to calculate permutation importance.
    - y_test: True labels for the test data.
    - n_repeats: Number of times to shuffle a feature for importance calculation (default 10).
    - random_state: Random seed for reproducibility (default 42).
    """
    
    # Calculate permutation importance
    result = permutation_importance(best_model, X_test, y_test, n_repeats=n_repeats, random_state=random_state)
    
    # Create a dataframe to display feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,  # Assumes X_train is a DataFrame
        'Importance': result.importances_mean
    })
    
    # Sort by importance and display the result
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(sorted_df)
    
    model_name = best_model['Model Name']
    # Optional: Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_df['Feature'], sorted_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title(f'Feature Importance for {model_name} Model')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the most important at the top
    plt.show()



def evaluate_models(best_models, X_test, y_test, model_type='classification', scoring=None):
    """
    Evaluates the best models on the test set and displays a table with the model performance.
    If no scoring is set, all models are displayed. If a scoring is set, only the best model
    per type is displayed based on the chosen scoring metric. Creates charts based on model type.

    Args:
        best_models (list): List of dictionaries containing best models and associated metadata.
        X_test (pd.DataFrame or np.array): Test feature set.
        y_test (pd.Series or np.array): Test target values.
        model_type (str): The type of model ('classification' or 'regression').
        scoring (str): The metric to score models on ('accuracy', 'f1', 'precision', 'recall' for classification,
                      'rmse', 'r2', 'mae' for regression).

    Returns:
        None
    """
    if model_type not in ['classification', 'regression']:
        raise ValueError("model_type must be 'classification' or 'regression'")

    results_table = []

    # Evaluate each model
    for model_info in best_models:
        model = model_info['Best Pipeline']
        model_name = model_info['Model Name']
        best_params = model_info['Best Parameters']

        # Predictions on the test set
        y_pred = model.predict(X_test)

        if model_type == 'classification':
            # Calculate classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

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
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))

            results_table.append({
                'Model': model_name,
                'RMSE': rmse,
                'R²': r2,
                'MAE': mae,
                'Best Parameters': best_params
            })

    # Convert the results to a DataFrame
    df = pd.DataFrame(results_table)

    # If scoring is set, find and display only the best model based on the metric
    if scoring:
        if model_type == 'classification':
            if scoring == 'accuracy':
                best_model_df = df.loc[df['Accuracy'].idxmax()]
            elif scoring == 'precision':
                best_model_df = df.loc[df['Precision'].idxmax()]
            elif scoring == 'recall':
                best_model_df = df.loc[df['Recall'].idxmax()]
            elif scoring == 'f1':
                best_model_df = df.loc[df['F1 Score'].idxmax()]
            else:
                raise ValueError(f"Invalid scoring parameter for classification: {scoring}")
        else:
            if scoring == 'rmse':
                best_model_df = df.loc[df['RMSE'].idxmin()]  # For RMSE, lower is better
            elif scoring == 'r2':
                best_model_df = df.loc[df['R²'].idxmax()]
            elif scoring == 'mae':
                best_model_df = df.loc[df['MAE'].idxmin()]  # For MAE, lower is better
            else:
                raise ValueError(f"Invalid scoring parameter for regression: {scoring}")

        # Display the best model result using tabulate
        print("\nBest Model Evaluation Results:")
        print(tabulate(best_model_df.to_frame().T, headers='keys', tablefmt='grid', showindex=False, numalign='right', floatfmt=".4f"))
    else:
        # Display all model results if no scoring metric is provided
        print("\nAll Model Evaluation Results:")
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False, numalign='right', floatfmt=".4f"))

    # Generate appropriate charts based on the model type
    if model_type == 'classification':
        # Bar chart for classification models
        df.set_index('Model', inplace=True)
        df[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', figsize=(12, 6))
        plt.title('Classification Model Performance')
        plt.xlabel('Model')
        plt.ylabel('Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        # Bar chart for regression models
        df.set_index('Model', inplace=True)
        df[['RMSE', 'R²', 'MAE']].plot(kind='bar', figsize=(12, 6), color=['lightcoral', 'lightblue', 'lightgreen'])
        plt.title('Regression Model Performance')
        plt.xlabel('Model')
        plt.ylabel('Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Display hyperparameters in rows of 4
    print("\nModel Hyperparameters:")
    for i, param_dict in enumerate(best_models, start=1):
        model_name = param_dict['Model Name']
        params = param_dict['Best Parameters']
        print(f"{i}. {model_name}: {', '.join([f'{k}={v}' for k, v in params.items()])}")