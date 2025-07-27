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
        ner_model = pipeline("ner", model=model_name)
        logger.info("NER model loaded successfully.")
        return ner_model
    except Exception as e:
        logger.error(f"An error occurred while loading the NER model: {e}")
        raise

def extract_diseases(ner_model, text, max_length=512):
    try:
        # Truncate the text to the maximum length the model can handle
        truncated_text = text[:max_length]
        entities = ner_model(truncated_text)

        # Check if 'entity_group' or 'entity' is the correct key
        diseases = list(set(
            entity['word'] for entity in entities
            if ('entity_group' in entity and entity['entity_group'] == "DISEASE") or
            ('entity' in entity and entity['entity'].endswith('DIS'))
        ))
        logger.info("Disease extraction completed successfully.")
        return diseases
    except Exception as e:
        logger.error(f"An error occurred during disease extraction: {e}")
        raise

def process_abstracts(df, text_column="abstract", id_column="abstract_id"):
    try:
        classifier = load_multi_label_classifier()
        ner_model = load_ner_model()
        results = []
        for _, row in df.iterrows():
            abstract_id = row[id_column]
            text = row[text_column]
            classification_result = predict_multi_label(classifier, text, ["Cancer", "Non-Cancer"])
            diseases = extract_diseases(ner_model, text)
            results.append({
                id_column: abstract_id,
                "predicted_categories": classification_result["predicted_labels"],
                "confidence_scores": classification_result["confidence_scores"],
                "extracted_diseases": diseases
            })
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"An error occurred during abstract processing: {e}")
        raise
