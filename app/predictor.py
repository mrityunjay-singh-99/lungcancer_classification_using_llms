import logging
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path, tokenizer_path):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Model and tokenizer loaded successfully from {model_path}.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"An error occurred while loading the model or tokenizer: {e}")
        raise

def preprocess_data(texts, tokenizer, max_length=512):
    try:
        inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        logger.info("Data preprocessing completed successfully.")
        return inputs
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        raise

def predict(model, inputs):
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logger.info("Predictions completed successfully.")
        return outputs.logits
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise

def postprocess_predictions(logits):
    try:
        predictions = torch.argmax(logits, dim=1).tolist()
        logger.info("Postprocessing of predictions completed successfully.")
        return predictions
    except Exception as e:
        logger.error(f"An error occurred during postprocessing: {e}")
        raise

def save_predictions(predictions, directory):
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "predictions.txt")
        with open(file_path, 'w') as f:
            for pred in predictions:
                f.write(f"{pred}\n")
        logger.info(f"Predictions saved successfully to {directory}.")
    except Exception as e:
        logger.error(f"An error occurred while saving predictions: {e}")
        raise
