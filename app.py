from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')  # The main form page

@app.route('/predict_web', methods=['POST'])
def predict_datapoint():
    try:
        # Collect data from form
        data = CustomData(
            age=int(request.form.get('age')),
            gender=request.form.get('gender'),
            education_level=request.form.get('education_level'),
            annual_income=float(request.form.get('annual_income')),
            employment_experience_years=float(request.form.get('employment_experience_years')),
            home_ownership_status=request.form.get('home_ownership_status'),
            loan_amount=float(request.form.get('loan_amount')),
            loan_purpose=request.form.get('loan_purpose'),
            interest_rate=float(request.form.get('interest_rate')),
            loan_to_income_ratio=float(request.form.get('loan_to_income_ratio')),
            credit_history_length_years=float(request.form.get('credit_history_length_years')),
            credit_score=int(request.form.get('credit_score')),
            prior_default_flag=int(request.form.get('prior_default_flag'))
        )

        # Convert to dataframe
        pred_df = data.get_data_as_data_frame()

        # Predict
        predict_pipeline = PredictPipeline()
        pred_class, pred_prob = predict_pipeline.predict(pred_df)

        # Map class to label
        pred_label = "Accepted" if pred_class[0] == 0 else "Rejected"
        prob_percent = (pred_prob[0])

        return render_template(
            'home.html',
            prediction=pred_class[0],
            status=pred_label,
            probability=prob_percent
        )

    except Exception as e:
        return render_template('home.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)