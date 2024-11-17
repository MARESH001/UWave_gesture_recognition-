import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.pipeline.exception import CustomException

def save_object(file_path, obj):
    """
    Function to save an object to a file using pickle.
    Args:
    - file_path: Path where the object will be saved.
    - obj: The object to be saved.

    Raises:
    - CustomException: If an error occurs during the save operation.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f"Error saving object: {e}", sys)

def load_object(file_path):
    """
    Function to load an object from a file using pickle.
    Args:
    - file_path: Path of the file from which the object will be loaded.

    Returns:
    - The loaded object.
    
    Raises:
    - CustomException: If an error occurs during the load operation.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(f"Error loading object: {e}", sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Function to evaluate multiple models using GridSearchCV for hyperparameter tuning.
    Args:
    - X_train: Training feature set.
    - y_train: Training target set.
    - X_test: Testing feature set.
    - y_test: Testing target set.
    - models: Dictionary of models to be evaluated.
    - param: Dictionary of hyperparameters for each model.

    Returns:
    - A dictionary of models with their test R2 scores.

    Raises:
    - CustomException: If an error occurs during model evaluation.
    """
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})

            # Apply GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the best parameters found by GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 score for train and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(f"Error evaluating models: {e}", sys)
