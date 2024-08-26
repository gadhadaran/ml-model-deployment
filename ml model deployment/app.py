from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model once when the server starts
try:
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model is not available'}), 500

    try:
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        age = float(request.form['age'])

        # Ensure inputs are within expected ranges (just as an example of basic validation)
        if not (0 <= glucose <= 300 and 0 <= bmi <= 100 and 0 <= age <= 120):
            return jsonify({'error': 'Input values out of range'}), 400

        # Make the prediction
        features = np.array([[glucose, bmi, age]])
        prediction = model.predict(features)[0]

        result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"

        return jsonify({'result': result})

    except ValueError:
        return jsonify({'error': 'Invalid input. Please ensure that all inputs are numeric.'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(debug=True)
