import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_predictions(file_path):
    try:
        with open(file_path, 'r') as f:
            predictions = [int(line.strip()) for line in f.readlines()]
        logger.info(f"Predictions loaded successfully from {file_path}.")
        return predictions
    except Exception as e:
        logger.error(f"An error occurred while loading predictions: {e}")
        raise

def evaluate_model(y_true, y_pred):
    try:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        logger.info("Model evaluation completed successfully.")
        return acc, f1, cm
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        raise

def print_confusion_matrix(cm):
    try:
        print("\nConfusion Matrix:")
        print("               Predicted Cancer   Predicted Non-Cancer")
        print(f"Actual Cancer        {cm[1][1]}                    {cm[1][0]}")
        print(f"Actual Non-Cancer   {cm[0][1]}                    {cm[0][0]}")
        logger.info("Confusion matrix printed successfully.")
    except Exception as e:
        logger.error(f"An error occurred while printing the confusion matrix: {e}")
        raise

def print_evaluation_metrics(acc, f1):
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")

def compare_model_performance(acc_baseline, f1_baseline, acc_finetune, f1_finetune):
    print("\nPerformance Comparison:")
    print(f"Accuracy Improvement: {acc_finetune - acc_baseline:.4f}")
    print(f"F1-Score Improvement: {f1_finetune - f1_baseline:.4f}")
