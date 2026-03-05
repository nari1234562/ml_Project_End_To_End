import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Preprocess features
            data_scaled = preprocessor.transform(features)

            # Prediction and probability
            preds = model.predict(data_scaled)
            probs = model.predict_proba(data_scaled)[:, 1]  # probability of rejection

            return preds, probs

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, age: int, gender: str, education_level: str, annual_income: float,
                 employment_experience_years: int, home_ownership_status: str,
                 loan_amount: float, loan_purpose: str, interest_rate: float,
                 loan_to_income_ratio: float, credit_history_length_years: int,
                 credit_score: int, prior_default_flag: int):

        self.age = age
        self.gender = gender
        self.education_level = education_level
        self.annual_income = annual_income
        self.employment_experience_years = employment_experience_years
        self.home_ownership_status = home_ownership_status
        self.loan_amount = loan_amount
        self.loan_purpose = loan_purpose
        self.interest_rate = interest_rate
        self.loan_to_income_ratio = loan_to_income_ratio
        self.credit_history_length_years = credit_history_length_years
        self.credit_score = credit_score
        self.prior_default_flag = prior_default_flag

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "education_level": [self.education_level],
                "annual_income": [self.annual_income],
                "employment_experience_years": [self.employment_experience_years],
                "home_ownership_status": [self.home_ownership_status],
                "loan_amount": [self.loan_amount],
                "loan_purpose": [self.loan_purpose],
                "interest_rate": [self.interest_rate],
                "loan_to_income_ratio": [self.loan_to_income_ratio],
                "credit_history_length_years": [self.credit_history_length_years],
                "credit_score": [self.credit_score],
                "prior_default_flag": [self.prior_default_flag]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)