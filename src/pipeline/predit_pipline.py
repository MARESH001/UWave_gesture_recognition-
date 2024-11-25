import os
import sys
import pandas as pd
from src.pipeline.utilis import load_object 
from  src.pipeline import loggers 
from src.pipeline.exception import ModelEvaluationError# Correct import statement

class PredictPipeline:
    def __init__(self):
        self.loggers = loggers.get_logger(__name__)

    def predict(self, features):
        try: 
            # Define the paths for the model and preprocessor
            model_path = os.path.join("artifacts", "best_model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            print("Before loading...")

            # Load the model and preprocessor using load_object (Make sure your load_object function works properly)
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After loading...")

            # Ensure the features are processed through the preprocessor before making predictions
            features_transformed = preprocessor.transform(features)  # Apply the transformation on input features
            print(f"Expected feature names: {model.feature_names_in_}")
            print(f"Provided feature names: {features_transformed.columns}")
            print("Checkpoint: Data transformed successfully.")
            print(f"Shape of input data: {features_transformed.shape}")

            # Use the model to get predictions
            predictions = model.predict(features_transformed)  # Assuming the model was trained using PyCaret
            print("Predictions generated...")

            return predictions
        
        except Exception as e:
            self.loggers.error(f"Error during model evaluation: {e}")
            raise ModelEvaluationError(f"Error during model evaluation: {e}", error_detail=sys)

