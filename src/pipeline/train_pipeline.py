import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        logging.info("Training Pipeline Started")

        # ==============================
        # 1️⃣ DATA INGESTION
        # ==============================
        logging.info("Starting Data Ingestion")

        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion Completed")

        # ==============================
        # 2️⃣ DATA VALIDATION
        # ==============================
        logging.info("Starting Data Validation")

        validation = DataValidation()
        validation_status = validation.validate_all_columns()

        if not validation_status:
            raise Exception("Data Validation Failed. Stopping Pipeline.")

        logging.info("Data Validation Passed")

        # ==============================
        # 3️⃣ DATA TRANSFORMATION
        # ==============================
        logging.info("Starting Data Transformation")

        data_transformation = DataTransformation()

        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )

        logging.info("Data Transformation Completed")

        # ==============================
        # 4️⃣ MODEL TRAINER
        # ==============================
        logging.info("Starting Model Training")

        model_trainer = ModelTrainer()

        best_model_score = model_trainer.initiate_model_trainer(
            train_array=train_arr,
            test_array=test_arr
        )

        logging.info(f"Model Training Completed. Best F1 Score: {best_model_score}")

        logging.info("Training Pipeline Completed Successfully")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()