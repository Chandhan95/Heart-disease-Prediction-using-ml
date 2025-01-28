from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
app = Flask(__name__)
# Load the trained KNN model
model = joblib.load('Heart-Prediction-KNN-Classifier.joblib')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_features = [float(x) for x in request.form.values()]
    if len(input_features) != 13:
        return jsonify({'error': 'Please provide exactly 13 input features.'})
     # Reshape input for the model
    input_data = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON or render the result
    return render_template('index.html', prediction_text=f'Heart Prediction (0 or 1): {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
