from transformers import pipeline
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_multi_label_classifier(model_name="facebook/bart-large-mnli"):
    try:
        classifier = pipeline("zero-shot-classification", model=model_name)
        logger.info("Multi-label classifier loaded successfully.")
        return classifier
    except Exception as e:
        logger.error(f"An error occurred while loading the classifier: {e}")
        raise

def predict_multi_label(classifier, text, candidate_labels):
    try:
        result = classifier(text, candidate_labels=candidate_labels)
        logger.info("Multi-label prediction completed successfully.")
        return {
            "predicted_labels": result['labels'],
            "confidence_scores": dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise

def load_ner_model(model_name="emilyalsentzer/Bio_ClinicalBERT"):
    try:
        ner_model = pipeline("ner", model=model_name, aggregation_strategy="simple")
        logger.info("NER model loaded successfully.")
        return ner_model
    except Exception as e:
        logger.error(f"An error occurred while loading the NER model: {e}")
        raise

def extract_diseases(ner_model, text, max_length=512):
    try:
        truncated_text = text[:max_length]
        entities = ner_model(truncated_text)
        diseases = list(set(entity['word'] for entity in entities if entity['entity_group'] == "DISEASE"))
        logger.info("Disease extraction completed successfully.")
        return diseases
    except Exception as e:
        logger.error(f"An error occurred during disease extraction: {e}")
        raise

def process_dataframe(df, text_column="combined_text"):
    try:
        if text_column not in df.columns:
            raise ValueError(f"The column '{text_column}' does not exist in the DataFrame.")

        classifier = load_multi_label_classifier()
        ner_model = load_ner_model()

        results = df[text_column].apply(lambda text: predict_multi_label(classifier, text, ["Cancer", "Non-Cancer"]))
        results_df = results.apply(pd.Series)
        df = pd.concat([df, results_df], axis=1)

        df["extracted_diseases"] = df[text_column].apply(lambda text: extract_diseases(ner_model, text))
        return df
    except Exception as e:
        logger.error(f"An error occurred during DataFrame processing: {e}")
        raise
