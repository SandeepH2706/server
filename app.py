from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the trained model
filename = 'sign_language_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions using the loaded model
        predictions = loaded_model.predict(df)
        
        # Prepare response
        response = {'predictions': predictions.tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
