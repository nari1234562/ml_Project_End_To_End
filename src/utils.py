import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a file using pickle.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param, threshold=0.6):
    """
    Evaluate multiple models with GridSearchCV and compute metrics using a custom threshold.
    Supports binary classification with optional thresholding for predicted probabilities.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        models: dictionary of model_name -> model_object
        param: dictionary of model_name -> hyperparameter grid
        threshold: float, probability threshold to convert predictions to 0/1

    Returns:
        report: dictionary containing train and test metrics for each model
    """
    try:
        report = {}

        for i, model_name in enumerate(models):
            model = list(models.values())[i]
            params = param[model_name]

            # Grid Search for best hyperparameters
            gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)

            # Train model on full training set
            model.fit(X_train, y_train)

            # Predictions with optional thresholding
            if hasattr(model, "predict_proba"):
                y_train_prob = model.predict_proba(X_train)[:, 1]
                y_test_prob = model.predict_proba(X_test)[:, 1]

                y_train_pred = (y_train_prob >= threshold).astype(int)
                y_test_pred = (y_test_prob >= threshold).astype(int)
            else:
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                y_train_prob = y_train_pred
                y_test_prob = y_test_pred

            # Compute metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_auc = roc_auc_score(y_train, y_train_prob)

            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_prob)

            # Store metrics
            report[model_name] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_precision": train_precision,
                "test_precision": test_precision,
                "train_recall": train_recall,
                "test_recall": test_recall,
                "train_f1": train_f1,
                "test_f1": test_f1,
                "train_auc": train_auc,
                "test_auc": test_auc
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)