import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
superkart_api = Flask("superkart_sales_api")

# Load the trained Boston housing model
model = joblib.load("/content/drive/MyDrive/p7_ModelDeployment/backend_files/superkart_sales_forecast_model_v1_0.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the superkart sales Prediction API!"

# Define an endpoint to predict SuperKart sales
@superkart_api.post('/v1/predict')
def predict_sales():
  try:
    # Get JSON data from the request
    data  = request.get_json()
    print("Raw incoming data:", data)

    # Validate expected fields
    required_fields = [
        'Product_Weight',
        'Product_Sugar_Content',
        'Product_Allocated_Area',
        'Product_MRP',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type',
        'Store_Age_Years',
        'Product_Type_Category'
    ]
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        return jsonify({'error': f"Missing fields: {missing_fields}"}), 400

    # Convert and transform input
    sample = {
        'Product_Weight': float(data['Product_Weight']),
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area_Log': np.log1p(float(data['Product_Allocated_Area'])),
        'Product_MRP': float(data['Product_MRP']),
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Store_Age_Years': int(data['Store_Age_Years']),
        'Product_Type_Category': data['Product_Type_Category']
    }


    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Predicted_Sales': prediction})
  except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
