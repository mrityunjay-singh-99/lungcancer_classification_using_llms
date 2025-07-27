from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from data_loader import load_and_parse_data
from data_preprocessing import load_and_clean_data
from model_baseline import predict_multi_label, load_multi_label_classifier
from disease_extractor import extract_diseases, load_ner_model
import logging
import os

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and preprocess data
input_df = load_and_parse_data()
output_directory = "preprocessed_data"
output_file_name = "cleaned_and_encoded_pubmed_data.csv"
output_full_path = os.path.join(output_directory, output_file_name)

cleaned_file_path = load_and_clean_data(
    input_df.copy(),
    output_full_path,
    text_columns=['title', 'abstract'],
    label_column='label',
    id_column='abstract_id',
    columns_to_keep=['abstract_id', 'title', 'abstract', 'label', 'label_encoded', 'combined_text']
)

if os.path.exists(cleaned_file_path):
    loaded_df = pd.read_csv(cleaned_file_path)
    logger.info(f"Loaded Cleaned DataFrame from '{cleaned_file_path}'.")
else:
    logger.error(f"Error: File not found at {cleaned_file_path}")

# Load models
classifier = load_multi_label_classifier()
ner_model = load_ner_model()

class TextRequest(BaseModel):
    text: str
    abstract_id: str = "unknown"

@app.post("/predict")
async def predict(request: TextRequest):
    try:
        text = request.text
        abstract_id = request.abstract_id

        # Predict categories and confidence scores
        prediction_result = predict_multi_label(classifier, text, ["Cancer", "Non-Cancer"])
        predicted_labels = prediction_result["predicted_labels"]
        confidence_scores = prediction_result["confidence_scores"]

        # Extract diseases
        diseases = extract_diseases(ner_model, text)

        # Prepare the response
        response = {
            "predicted_categories": {
                "predicted_labels": predicted_labels,
                "confidence_scores": confidence_scores
            },
            "disease_identification": {
                "abstract_id": abstract_id,
                "extracted_diseases": diseases
            }
        }

        return response
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
