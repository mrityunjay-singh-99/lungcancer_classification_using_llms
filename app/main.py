from data_loader import load_and_parse_data
from data_preprocessing import load_and_clean_data
from model_baseline import process_dataframe as process_dataframe_baseline
from model_finetune import apply_fine_tuning
from disease_extractor import process_abstracts
from train_test_split import kfold_cross_validation
from evaluation import evaluate_model, compare_model_performance, load_predictions, print_evaluation_metrics, print_confusion_matrix
from predictor import load_model_and_tokenizer, preprocess_data, predict, postprocess_predictions, save_predictions
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
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
        return

    # # Baseline model predictions
    # processed_df_baseline = process_dataframe_baseline(loaded_df, text_column="combined_text")
    # logger.info("Processed DataFrame with baseline model predictions.")

    # # Fine-tuning
    # apply_fine_tuning(loaded_df)
    # logger.info("Fine-tuning applied.")

    # Disease extraction
    processed_df_disease = process_abstracts(loaded_df)
    logger.info("Disease extraction completed.")

    # K-fold cross-validation
    folds = kfold_cross_validation(loaded_df)
    train_df, test_df = folds[0]
    logger.info("K-fold cross-validation completed.")

    # Evaluation
    y_true = [0, 1, 0, 0, 1, 1]
    y_pred_baseline = load_predictions("model_baseline/predictions.txt")
    y_pred_finetune = load_predictions("model_finetune/predictions.txt")

    acc_baseline, f1_baseline, cm_baseline = evaluate_model(y_true, y_pred_baseline)
    print("Baseline Model Evaluation:")
    print_evaluation_metrics(acc_baseline, f1_baseline)
    print_confusion_matrix(cm_baseline)

    acc_finetune, f1_finetune, cm_finetune = evaluate_model(y_true, y_pred_finetune)
    print("\nFine-Tuned Model Evaluation:")
    print_evaluation_metrics(acc_finetune, f1_finetune)
    print_confusion_matrix(cm_finetune)

    compare_model_performance(acc_baseline, f1_baseline, acc_finetune, f1_finetune)

    # Predictions
    baseline_model_path = "path_to_baseline_model"
    finetune_model_path = "path_to_finetune_model"
    tokenizer_path = "path_to_tokenizer"

    baseline_model, tokenizer = load_model_and_tokenizer(baseline_model_path, tokenizer_path)
    finetune_model, _ = load_model_and_tokenizer(finetune_model_path, tokenizer_path)

    texts = loaded_df["combined_text"].tolist()
    inputs = preprocess_data(texts, tokenizer)

    baseline_logits = predict(baseline_model, inputs)
    baseline_predictions = postprocess_predictions(baseline_logits)
    save_predictions(baseline_predictions, "model_baseline")

    finetune_logits = predict(finetune_model, inputs)
    finetune_predictions = postprocess_predictions(finetune_logits)
    save_predictions(finetune_predictions, "model_finetune")

if __name__ == "__main__":
    main()
