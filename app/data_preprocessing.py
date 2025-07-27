import pandas as pd
import re
import string
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

def clean_text(text, remove_punctuation=True, remove_digits=True, normalize_citations=True):
    if not isinstance(text, str):
        logger.debug(f"Non-string input for clean_text: {type(text)}. Returning empty string.")
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    if normalize_citations:
        text = re.sub(r'\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]', '', text)
        text = re.sub(r'\(\s*[A-Za-z]+(?: et al\.)?,\s*\d{4}\s*\)', '', text)
        text = re.sub(r'[A-Za-z]+(?: et al\.)?\s*\(\d{4}\)', '', text)
        text = re.sub(r'(?:ref\.?|refs\.?|references?|see|citation)\s*\[\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*\]', '', text)
    if remove_punctuation:
        text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    if remove_digits:
        text = re.sub(r"\d+", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def encode_labels(df, column_name):
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found for encoding. Skipping label encoding.")
        return df
    try:
        le = LabelEncoder()
        df[column_name + "_encoded"] = le.fit_transform(df[column_name])
        logger.info(f"Labels in column '{column_name}' encoded successfully.")
        return df
    except Exception as e:
        logger.error(f"An error occurred during label encoding for column '{column_name}': {e}")
        raise

def load_and_clean_data(input_df, output_file, text_columns=['abstract'], label_column='label', id_column='abstract_id', columns_to_keep=None):
    if not isinstance(input_df, pd.DataFrame):
        logger.error("Input must be a pandas DataFrame.")
        raise TypeError("Input must be a pandas DataFrame.")
    if not isinstance(output_file, str) or not output_file:
        logger.error("Output file path must be a non-empty string.")
        raise ValueError("Output file path must be a non-empty string.")

    df = input_df.copy()
    try:
        if columns_to_keep is None:
            columns_to_keep = [id_column] + text_columns
            if label_column in df.columns:
                columns_to_keep.append(label_column)
            if label_column in df.columns:
                columns_to_keep.append(label_column + "_encoded")

        existing_cols_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_cols_to_keep]
        logger.info(f"DataFrame filtered to keep columns: {existing_cols_to_keep}")

        if id_column in df.columns:
            df['abstract_id'] = df["abstract_id"].apply(lambda x: re.sub(r'[^0-9]', '', x))
            existing_cols_to_keep.append('abstract_id')
            logger.info(f"Created 'abstract_id' column from '{id_column}'.")

        for col in text_columns:
            if col in df.columns:
                df[col].fillna('', inplace=True)
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(clean_text)
                logger.info(f"Column '{col}' cleaned and missing values handled.")
            else:
                logger.warning(f"Text column '{col}' not found in DataFrame. Skipping cleaning for this column.")

        if label_column in df.columns:
            df = encode_labels(df, label_column)
        else:
            logger.warning(f"Label column '{label_column}' not found in DataFrame. Skipping label encoding.")

        if len(text_columns) > 1:
            df['combined_text'] = df[text_columns[0]]
            for i in range(1, len(text_columns)):
                df['combined_text'] = df['combined_text'] + " " + df[text_columns[i]]
            df['combined_text'] = df['combined_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            logger.info("Created 'combined_text' column from multiple text columns.")
        elif len(text_columns) == 1:
            df['combined_text'] = df[text_columns[0]]
            logger.info(f"Single text column '{text_columns[0]}' used as 'combined_text'.")
        else:
            logger.warning("No text columns specified for cleaning or combination.")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        df.to_csv(output_file, index=False)
        logger.info(f"Data cleaned, processed, and saved successfully to {output_file}.")
        return output_file
    except Exception as e:
        logger.error(f"An error occurred in load_and_clean_data: {e}")
        raise
