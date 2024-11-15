import os
import pandas as pd
from src.loggers import get_logger

# Initialize the logger
logger = get_logger(__name__)

class DataTransformation:
    """Handles data transformation tasks such as checking for non-numeric values and splitting features and target."""
    
    def __init__(self, artifacts_dir='artifacts'):
        self.artifacts_dir = artifacts_dir
        self.train_data_path = os.path.join(self.artifacts_dir, 'train.csv')
        self.test_data_path = os.path.join(self.artifacts_dir, 'test.csv')
    
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
    
    def check_non_numeric_values(self, train_df, test_df):
        """Checks for non-numeric values in the DataFrames."""
        # Check for non-numeric values in the training data
        non_numeric_train = train_df.applymap(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit())
        non_numeric_test = test_df.applymap(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit())

        logger.info("Checking for non-numeric values in the training data...")
        non_numeric_train_rows = train_df[non_numeric_train.any(axis=1)]
        logger.info(f"Non-numeric values in training set:")
        logger.info(non_numeric_train_rows)

        logger.info("Checking for non-numeric values in the test data...")
        non_numeric_test_rows = test_df[non_numeric_test.any(axis=1)]
        logger.info(f"Non-numeric values in test set:")
        logger.info(non_numeric_test_rows)
        
        # Returning non-numeric rows for further handling if needed
        return non_numeric_train_rows, non_numeric_test_rows
    
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

            # Check for non-numeric values in the data
            non_numeric_train_rows, non_numeric_test_rows = self.check_non_numeric_values(train_df, test_df)

            # Split data into features and target
            X_train, y_train, X_test, y_test = self.split_features_and_target(train_df, test_df)

            # Returning all the transformed data for future use
            return X_train, y_train, X_test, y_test, non_numeric_train_rows, non_numeric_test_rows
        except Exception as e:
            logger.error(f"An error occurred during data transformation: {e}")
            raise

# Usage example
if __name__ == "__main__":
    data_transformation = DataTransformation()

    try:
        # Perform data transformation
        X_train, y_train, X_test, y_test, non_numeric_train, non_numeric_test = data_transformation.transform_data()

        # Optionally, save the results to CSV for further inspection
        non_numeric_train.to_csv('non_numeric_train.csv', index=False)
        non_numeric_test.to_csv('non_numeric_test.csv', index=False)

        logger.info("Non-numeric rows saved to 'non_numeric_train.csv' and 'non_numeric_test.csv'")

        # You can also save the transformed data (features and target)
        X_train.to_csv('X_train.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)

        logger.info("Transformed data saved to 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv'")

    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        print(f"Data transformation failed: {e}")
