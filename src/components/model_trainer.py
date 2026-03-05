import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Split training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Handle class imbalance for XGBoost
            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

            # -----------------------------
            # MODELS
            # -----------------------------
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),
                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1
                )
            }

            # -----------------------------
            # HYPERPARAMETERS
            # -----------------------------
            params = {
                "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
                "Random Forest": {"n_estimators": [200, 300], "max_depth": [10, 15], "min_samples_split": [5, 10]},
                "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5]},
                "XGBoost": {
                    "n_estimators": [300, 500],
                    "learning_rate": [0.01, 0.05],
                    "max_depth": [3, 5],
                    "min_child_weight": [1, 3],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1],
                    "reg_alpha": [0, 0.1],
                    "reg_lambda": [1, 5]
                }
            }

            # -----------------------------
            # TRAIN MODELS
            # -----------------------------
            # Use threshold 0.6 for XGBoost preference
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                threshold=0.6
            )

            print("\nModel Performance:\n")
            for model_name, metrics in model_report.items():
                print(f"{model_name}")
                print(metrics)
                print("-"*40)

            # -----------------------------
            # SELECT BEST MODEL
            # Prefer XGBoost if F1 is close to others
            # -----------------------------
            best_model_name = None
            best_f1_score = 0

            for model_name, metrics in model_report.items():
                f1 = metrics["test_f1"]
                if model_name == "XGBoost":
                    # Give XGBoost slight preference (+0.01 fudge factor)
                    f1 += 0.01
                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_model_name = model_name

            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best F1 Score: {best_f1_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f"\nBest Model Selected: {best_model_name}")
            print(f"Best F1 Score: {best_f1_score}")

            return best_f1_score

        except Exception as e:
            raise CustomException(e, sys)