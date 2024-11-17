import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend like Agg
import logging
from pycaret.classification import setup, create_model, evaluate_model, plot_model, save_model, tune_model, compare_models
import pandas as pd
import os
from src.pipeline.loggers import get_logger  # Assuming logger is in src.loggers
from src.pipeline.exception import ModelTrainingError, ModelEvaluationError  # Import your custom exceptions

# Initialize logger
logger = get_logger(__name__)

# Set directory path where data is stored
data_dir = r'C:\Users\HP\Desktop\UWAVE_GESTURE_RECOGNITION\artifacts'


# Load the training and testing data (assuming CSV files for this example)
try:
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

    logger.info("Data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise ModelTrainingError(f"Error loading data: {e}", error_detail=str(e))

# Ensure the target column ('col_943') is in the correct format (e.g., numeric or categorical)
y_train = y_train.squeeze()  # If it's a DataFrame with a single column, convert it to a Series
y_test = y_test.squeeze()    # Same for test target

# Add the target variable column to both train_df and test_df
train_df = X_train.copy()
train_df['col_943'] = y_train  # Add target column

test_df = X_test.copy()
test_df['col_943'] = y_test  # Add target column

# Reset index of both train_df and test_df before passing to setup
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Initialize the setup in PyCaret
try:
    setup(data=train_df, target='col_943', session_id=123)
    logger.info("PyCaret setup initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing PyCaret setup: {e}")
    raise ModelTrainingError(f"Error initializing PyCaret setup: {e}")

# Optional: Compare models
try:
    best_model = compare_models()  # Automatically compares models and selects the best one
    logger.info(f"Best model: {best_model}")
except Exception as e:
    logger.error(f"Error comparing models: {e}")
    raise ModelTrainingError(f"Error comparing models: {e}")

# Create and tune the model
try:
    tuned_model = tune_model(best_model)  # Tune the best model
    logger.info(f"{best_model} tuned successfully.")
except Exception as e:
    logger.error(f"Error tuning model: {e}")
    raise ModelTrainingError(f"Error tuning model: {e}")

# Evaluate the model
try:
    evaluate_model(tuned_model)
    logger.info("Model evaluation completed successfully.")
except Exception as e:
    logger.error(f"Error during model evaluation: {e}")
    raise ModelEvaluationError(f"Error during model evaluation: {e}")

# Save the model
# Save the model
try:
    model_save_path = os.path.join(data_dir, 'best_model')  # Specify path in the 'artifacts' folder
    save_model(tuned_model, model_save_path)  # Save the model in the specified directory
    logger.info(f"Best model saved successfully at {model_save_path}.")
except Exception as e:
    logger.error(f"Error saving the model: {e}")
    raise ModelTrainingError(f"Error saving the model: {e}")

# Plot and save confusion matrix and ROC curve
try:
    plot_model(tuned_model, plot='confusion_matrix', plot_kwargs={'percent': True}, save=True)
    plot_model(tuned_model, plot='auc', save=True)
    logger.info("Confusion matrix and ROC curve plotted successfully.")
except Exception as e:
    logger.error(f"Error generating plots: {e}")
    raise ModelEvaluationError(f"Error generating plots: {e}")
