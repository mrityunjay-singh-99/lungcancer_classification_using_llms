import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_parse_data(base_dir="Dataset", labels=["Cancer", "Non-Cancer"]):
    data = []
    for label in labels:
        folder_path = os.path.join(base_dir, label)
        if not os.path.exists(folder_path):
            logger.warning(f"Directory not found: {folder_path}")
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    lines = content.split("\n")
                    if len(lines) < 3:
                        raise ValueError(f"File does not contain enough lines: {file_path}")
                    abstract_id = lines[0].replace('<ID:', '').replace('>', '').strip()
                    title = lines[1].replace('Title:', '').strip()
                    abstract = lines[2].replace('Abstract:', '').strip()
                    data.append({
                        "file_path": file_path,
                        "label": label,
                        "abstract_id": abstract_id,
                        "title": title,
                        "abstract": abstract,
                        "content": content
                    })
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
    return pd.DataFrame(data)
