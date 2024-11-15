import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from src.loggers import get_logger
from src.exception import DataIngestionError

# Initialize the logger
logger = get_logger(__name__)

class DataIngestionConfig:
    """Configuration for data ingestion specifying paths for train and test data."""
    artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)  # Create artifacts directory if it doesn't exist
    train_data_path: str = os.path.join(artifacts_dir, "train.csv")
    test_data_path: str = os.path.join(artifacts_dir, "test.csv")

class DataIngestion:
    """Handles the data ingestion process."""
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def load_arff_file(self, filepath, save_as_train=True):
        """
        Loads an ARFF file, processes it, and optionally saves it as a CSV.

        Args:
            filepath (str): The path to the ARFF file.
            save_as_train (bool): If True, saves the data as the training set, else as the test set.

        Returns:
            str: The path to the saved CSV file.
        """
        try:
            # Read the ARFF file content
            with open(filepath, 'r') as file:
                content = file.readlines()
            logger.info(f"ARFF file successfully read from {filepath}")

            # Find the start of the data section
            data_start_index = content.index('@data\n') + 1

            # Extract attribute names
            attribute_lines = [line for line in content[:data_start_index - 1] if line.startswith('@attribute')]
            attribute_names = [line.split()[1] for line in attribute_lines if not line.startswith('@attribute relationalAtt')]

            # Extract the data
            data_lines = content[data_start_index:]
            data = []
            max_cols = 0
            for line in data_lines:
                values = [v.strip() for v in line.strip().split(',')]
                if values and values[0] != '':
                    data.append(values)
                    max_cols = max(max_cols, len(values))

            if len(attribute_names) != max_cols:
                attribute_names = [f'col_{i}' for i in range(max_cols)]

            df = pd.DataFrame(data, columns=attribute_names)

            # Convert columns to numeric where possible
            for col in attribute_names:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

            logger.info("DataFrame successfully created from ARFF file")

            # Determine save path
            save_path = (self.ingestion_config.train_data_path if save_as_train else self.ingestion_config.test_data_path)
            df.to_csv(save_path, index=False, header=True)
            logger.info(f"DataFrame saved as CSV at {save_path}")

            return save_path

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise DataIngestionError(f"File not found: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise DataIngestionError(f"An unexpected error occurred during data ingestion: {e}")

# Usage example
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    try:
        # Load training data
        train_data_path = data_ingestion.load_arff_file('C:\\Users\\HP\\Desktop\\UWAVE_GESTURE_RECOGNITION\\notebook\\data\\UWaveGestureLibrary_TRAIN.arff', save_as_train=True)
        logger.info(f"Training data saved at {train_data_path}")

        # Load test data
        test_data_path = data_ingestion.load_arff_file('C:\\Users\\HP\\Desktop\\UWAVE_GESTURE_RECOGNITION\\notebook\\data\\UWaveGestrueLibrary_TEST.arff', save_as_train=False)
        logger.info(f"Test data saved at {test_data_path}")

    except DataIngestionError as e:
        logger.error(e)
        print(e)
