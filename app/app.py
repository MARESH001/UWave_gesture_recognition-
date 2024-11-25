from flask import Flask, request, jsonify
import pandas as pd
from src.components.data_transformation import DataCleaningTransformer  # Adjust path if needed
from src.pipeline.predit_pipline import PredictPipeline  # Import the PredictPipeline class

app = Flask(__name__)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/')
def home():
    return "Welcome to the Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure data is received as JSON
        if request.is_json:
            data = request.get_json()
        else:
            return jsonify({"error": "Invalid input, expected JSON format"}), 400

        # Check if 'features' key exists in the data
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in the input data"}), 400

        features = data['features']
        num_features = 943 # Adjust based on your model's expected input size

        # Check the number of features matches the model's requirements
        if len(features) != num_features:
            return jsonify({"error": f"Expected {num_features} features, but got {len(features)}"}), 400

        # Convert the features to a DataFrame with expected column names
        column_names = [f'col_{i}' for i in range(num_features)]
        input_df = pd.DataFrame([features], columns=column_names)

        # Use the prediction pipeline to make predictions
        predictions = predict_pipeline.predict(input_df)
        if predictions is None:
            return jsonify({"error": "Prediction failed"}), 500

        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        # Handle any errors and return a proper error message
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
