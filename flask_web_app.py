import pickle
from flask import Flask, request, render_template
import numpy as np

web_app = Flask(__name__)
app = web_app

# Load the SVM model and standard scaler
svm_model = pickle.load(open('svm.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Collecting form data
            data = [
                float(request.form.get("age")),
                float(request.form.get("bloodPressure")),
                float(request.form.get("specificGravity")),
                float(request.form.get("albumin")),
                float(request.form.get("sugar")),
                1.0 if request.form.get("redBloodCells") == "normal" else 0.0,
                float(request.form.get("hemoglobin")),
                float(request.form.get("packedCellVolume"))
            ]
            
            # Convert to NumPy array and standardize
            input_data = np.array(data).reshape(1, -1)
            scaled_data = standard_scaler.transform(input_data)
            
            # Make a prediction
            prediction = svm_model.predict(scaled_data)
            
            # Interpret the result
            result = "No Chronic Kidney Disease" if prediction[0] == 0 else "Chronic Kidney Disease Detected"
            
            # Return result to the user
            return render_template('home.html', prediction_text=result)
        except Exception as e:
            # Handle any unexpected errors
            return render_template('home.html', prediction_text=f"Error: {str(e)}")
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
