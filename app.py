from flask import Flask, render_template, request, session, redirect, jsonify
from auth import auth_bp
from database import init_db
from flask_session import Session
from config import Config
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import json

# Load model metrics from static/model_metrics.json
with open("static/model_metrics.json", "r") as f:
    model_metrics = json.load(f)




app = Flask(__name__)
app.config.from_object(Config)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

init_db(app)

app.register_blueprint(auth_bp)

# Load the trained model
#model = joblib.load("./models/ensemble_model.pkl")  # Ensure this file exists
# Load multiple trained models
models = {
    "ensemble": joblib.load("./models/ensemble_model.pkl"),  # Ensure these files exist
    "random_forest": joblib.load("./models/rf_model.pkl"),
    "xgboost": joblib.load("./models/xgb_model.pkl"),
    "svm": joblib.load("./models/svm_model.pkl"),
    "knn":joblib.load("./models/knn_model.pkl") ,
    "mlpnn":joblib.load("./models/mlp_model.pkl"),
    # "lgboost":joblib.load("./models/lgb_model.pkl")
}
@app.route("/")
def home():
    return render_template("home.html")

@app.route('/dashboard')
# @login_required
def dashboard():
    with open('static/model_metrics.json') as f:
        metrics = json.load(f)

    # Add image paths for ROC curves
    roc_images = {model: f"static/roc_curves/{model}.png" for model in metrics.keys()}

    return render_template("dashboard.html", username=session["username"], models=models.keys(), metrics=metrics, roc_images=roc_images)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form
        # Get selected model
        selected_model = data["model"]
        if selected_model not in models:
            return jsonify({"error": "Invalid model selection"})

        selected_model = models[selected_model]
        # Convert form data to a numerical feature array
        features = np.array([
            float(data["age"]),
            float(data["total_bilirubin"]),
            float(data["direct_bilirubin"]),
            float(data["alkphos"]),
            float(data["sgpt"]),
            float(data["sgot"]),
            float(data["total_proteins"]),
            float(data["albumin"]),
            float(data["ag_ratio"]),
            int(data["gender_female"]),
            int(data["gender_male"])
        ]).reshape(1, -1)

        # Perform prediction
        prediction = selected_model.predict(features)[0]
        probabilities = selected_model.predict_proba(features)[0] # get probabilities score 
        cirrhosis_prob = probabilities[1] # probability of cirrhosis .(class 1)
        severity_percentage = round(cirrhosis_prob * 100, 2) # convert to precentage
        
        # Plot probability distribution
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["No Cirrhosis", "Cirrhosis"], probabilities, color=["green", "red"])
        ax.set_xlabel("Prediction Outcome")
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")
        plt.tight_layout()
        
       # Convert plot to Base64
        img_io = io.BytesIO()
        plt.savefig(img_io, format="png")
        img_io.seek(0)
        plot_url = base64.b64encode(img_io.getvalue()).decode()

        plt.close(fig)  # Close plot to prevent memory issues

        
        result = f"Cirrhosis Detected ({severity_percentage}%)" if prediction == 1 else "No Cirrhosis"

        return jsonify({"prediction": result,"plot_url":plot_url,"severity": severity_percentage})

    except Exception as e:
        return jsonify({"error": str(e)})
    
    
    
    
# new code batch prediction
import pandas as pd
import re
import pandas as pd
import joblib
from flask import request, jsonify
import unicodedata
import re

# Utility function to clean header strings
def clean_string(s):
    s = unicodedata.normalize('NFKC', str(s))
    s = re.sub(r'[\u2000-\u206F\u2E00-\u2E7F\u115F\uFEFFᅠ]', '', s)
    s = s.replace('\u3000','').replace('\u00a0','').replace('\u200f','').strip()
    return s

import pandas as pd
import joblib
from flask import Flask, request, jsonify
import os
from fpdf import FPDF 
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['csvFile']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load CSV into DataFrame
        df = pd.read_csv(file)

        # Clean column names: remove leading/trailing spaces and invisible characters
        df.columns = df.columns.str.strip().str.replace(r'[\u200b\u00a0\u202f]', '', regex=True)

        # One-hot encode gender if needed
        if 'gender' in df.columns:
            df['Gender of the patient_Female'] = (df['gender'].str.lower() == 'female').astype(int)
            df['Gender of the patient_Male'] = (df['gender'].str.lower() == 'male').astype(int)
            df.drop('gender', axis=1, inplace=True)

        # Load selected model
        model = joblib.load('./models/ensemble_model.pkl')

        # Clean expected columns from model
        expected_cols = [col.strip().replace('\u200b', '').replace('\u00a0', '').replace('\ufeff', '') for col in model.feature_names_in_]

        print("Cleaned CSV columns:", list(df.columns))
        print("Cleaned model expected columns:", expected_cols)

        # Validate columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing required columns in the CSV: {missing_cols}'})

        X = df[expected_cols]

        # # Predict
        preds = model.predict(X)
        df['Prediction'] = preds

        # return jsonify({
        #     'columns': list(df.columns),
        #     'data': df.values.tolist()
        # })
        
        # download csv 
        # import os

        # # Add prediction column
        # df['Prediction'] = preds

        # Save the result to a CSV file
        
        # Generate PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)

        # Heading
        pdf.cell(0, 10, "Liver Cirrhosis Prediction Results", ln=True, align='C')
        pdf.ln(10)

        # Table header
        pdf.set_font("Arial", 'B', 12)
        col_width = pdf.w / (len(df.columns) + 1)
        for col in df.columns:
            pdf.cell(col_width, 10, col, border=1,align='C')
        pdf.ln()

        # Table rows
        pdf.set_font("Arial", '', 10)
        for i in range(len(df)):
            for item in df.iloc[i]:
                pdf.cell(col_width, 10, str(item), border=1, align='C')
            pdf.ln()

        # Save PDF
        output_path = os.path.join('static', 'batch_prediction_result.pdf')
        pdf.output(output_path)
        # output_path = os.path.join('static', 'batch_prediction_result.pdf')
        # df.to_csv(output_path, index=False)

        # Return download link
        download_link = f'/download_batch_result'
        return jsonify({ 'columns': list(df.columns),
             'data': df.values.tolist(),'message': 'Prediction completed.', 'download_link': download_link})
    
    except Exception as e:
        return jsonify({'error': str(e)})
   




if __name__ == '__main__':
    app.run(debug=True)




