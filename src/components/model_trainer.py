import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend like Agg
import logging
from pycaret.classification import setup, create_model, evaluate_model, plot_model, save_model, tune_model
import pandas as pd
import os
from src.loggers import get_logger  # Assuming logger is in src.loggers
from src.exception import ModelTrainingError, ModelEvaluationError  # Import your custom exceptions

# Initialize logger
logger = get_logger(__name__)

# Set directory path where data is stored
data_dir = '../UWAVE_GESTURE_RECOGNITION'

# Load the training and testing data (assuming CSV files for this example)
try:
    # Assuming the CSV files are named 'X_train.csv', 'y_train.csv', 'X_test.csv', and 'y_test.csv'
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

    logger.info("Data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise ModelTrainingError(f"Error loading data: {e}")

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

# Create the Extra Trees model
try:
    et = create_model('et')  # Extra Trees classifier
    logger.info("Extra Trees model created successfully.")
except Exception as e:
    logger.error(f"Error creating Extra Trees model: {e}")
    raise ModelTrainingError(f"Error creating Extra Trees model: {e}")

# Optional: Compare models if needed
try:
    tuned_et = tune_model(et)  # Extra Trees classifier
    logger.info("Extra Trees model tuned successfully.")
except Exception as e:
    logger.error(f"Error tuning Extra Trees model: {e}")
    raise ModelTrainingError(f"Error tuning Extra Trees model: {e}")

# Evaluate the Extra Trees model
try:
    evaluate_model(tuned_et)  # This will generate evaluation plots, including ROC curve
    logger.info("Model evaluation completed successfully.")
except Exception as e:
    logger.error(f"Error during model evaluation: {e}")
    raise ModelEvaluationError(f"Error during model evaluation: {e}")

# Optionally, save the model
try:
    save_model(tuned_et, 'extra_trees_model')
    logger.info("Extra Trees model saved successfully.")
except Exception as e:
    logger.error(f"Error saving the model: {e}")
    raise ModelTrainingError(f"Error saving the model: {e}")

# Plot the confusion matrix for the Extra Trees model
try:
    plot_model(tuned_et, plot='confusion_matrix', plot_kwargs={'percent': True},save=True)
    plot_model(tuned_et, plot='auc', save = True)
    logger.info("Confusion matrix and ROC plot completed successfully.")
except Exception as e:
    logger.error(f"Error generating confusion matrix plot: {e}")
    raise ModelEvaluationError(f"Error generating confusion matrix plot: {e}")

# Save the model again at the end (optional step)
try:
    save_model(et, 'extra_trees_model')
    logger.info("Extra Trees model saved successfully.")
except Exception as e:
    logger.error(f"Error saving the model: {e}")
    raise ModelTrainingError(f"Error saving the model: {e}")
