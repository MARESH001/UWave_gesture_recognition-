import os
import pandas as pd
import numpy as np
from src.pipeline.loggers import get_logger
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Initialize the logger
logger = get_logger(__name__)

class DataCleaningTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning and preprocessing data."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit the transformer (no fitting needed for this simple example)."""
        return self

    def transform(self, X):
        """Transform the data by cleaning non-numeric values."""
        logger.info("Cleaning non-numeric values...")
        
        # Replace non-numeric entries with NaN
        X_cleaned = X.applymap(lambda x: np.nan if isinstance(x, str) and not x.replace('.', '', 1).isdigit() else x)
        
        # Fill or handle NaN values (e.g., imputation if necessary)
        X_cleaned = X_cleaned.fillna(0)  # Replace NaN with 0 (customize as needed)
        
        logger.info("Non-numeric values cleaned.")
        return X_cleaned


class DataTransformation:
    """Handles data transformation tasks such as checking for non-numeric values and splitting features and target."""
    
    def __init__(self, artifacts_dir='artifacts'):
        self.artifacts_dir = artifacts_dir
        self.train_data_path = os.path.join(self.artifacts_dir, 'train.csv')
        self.test_data_path = os.path.join(self.artifacts_dir, 'test.csv')
        self.preprocessor_file = os.path.join(self.artifacts_dir, 'preprocessor.pkl')  # Define the preprocessor file path
        self.preprocessor = DataCleaningTransformer()
    def load_data(self):
        """Loads the train and test CSV files into DataFrames."""
        try:
            # Load training and testing data
            train_df = pd.read_csv(self.train_data_path)
            test_df = pd.read_csv(self.test_data_path)
            logger.info(f"Loaded training data from {self.train_data_path}")
            logger.info(f"Loaded test data from {self.test_data_path}")
            return train_df, test_df
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            raise

    def save_preprocessor(self):
        """Saves the fitted preprocessor to a .pkl file."""
        try:
            joblib.dump(self.preprocessor, self.preprocessor_file)
            logger.info(f"Preprocessor saved to {self.preprocessor_file}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {e}")
            raise
    
    
    
    def split_features_and_target(self, train_df, test_df, target_column='col_943'):
        """
        Splits the DataFrames into feature variables (X) and target variables (y).
        
        Args:
            train_df (DataFrame): Training DataFrame.
            test_df (DataFrame): Testing DataFrame.
            target_column (str): The column name for the target variable.
        
        Returns:
            X_train (DataFrame): Feature set for training.
            y_train (Series): Target variable for training.
            X_test (DataFrame): Feature set for testing.
            y_test (Series): Target variable for testing.
        """
        try:
            # Ensure 'col_943' is set as the target variable
            X_train = train_df.drop(columns=[target_column])  # Drop target column from training set
            y_train = train_df[target_column]  # Target column in training set

            X_test = test_df.drop(columns=[target_column])  # Drop target column from testing set
            y_test = test_df[target_column]  # Target column in testing set

            logger.info(f"Features and target separated for train and test data.")

            return X_train, y_train, X_test, y_test
        except KeyError as e:
            logger.error(f"Target column '{target_column}' not found in the DataFrame: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while splitting features and target: {e}")
            raise

    
    def transform_data(self):
        """Main method to execute the transformations."""
        try:
            # Load data
            train_df, test_df = self.load_data()
    
            logger.info("Applying data cleaning transformations...")
    
            # Track non-numeric rows before cleaning
            non_numeric_train_rows = train_df[train_df.applymap(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit())]
            non_numeric_test_rows = test_df[test_df.applymap(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit())]
    
            # Fit the transformer on training data
            self.preprocessor.fit(train_df)
    
            # Apply the cleaning transformations
            train_df_cleaned = self.preprocessor.transform(train_df)
            test_df_cleaned = self.preprocessor.transform(test_df)
    
            # Save the fitted preprocessor for future use
            self.save_preprocessor()
    
            # Split data into features and target
            X_train, y_train, X_test, y_test = self.split_features_and_target(train_df, test_df)
    
            # Returning all the transformed data for future use
            return X_train, y_train, X_test, y_test, non_numeric_train_rows, non_numeric_test_rows
    
        except Exception as e:
            logger.error(f"An error occurred during data transformation: {e}")
            raise
        _


# Usage example
if __name__ == "__main__":
    data_transformation = DataTransformation()

    try:
        # Perform data transformation
        X_train, y_train, X_test, y_test, non_numeric_train_rows, non_numeric_test_rows = data_transformation.transform_data()

        # Optionally, save the transformed data
        X_train.to_csv('X_train_cleaned.csv', index=False)
        y_train.to_csv('y_train_cleaned.csv', index=False)
        X_test.to_csv('X_test_cleaned.csv', index=False)
        y_test.to_csv('y_test_cleaned.csv', index=False)

        logger.info("Transformed data saved to 'X_train_cleaned.csv', 'y_train_cleaned.csv', 'X_test_cleaned.csv', 'y_test_cleaned.csv'")

    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        print(f"Data transformation failed: {e}")
