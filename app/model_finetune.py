import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tokenizer(model_name="distilbert-base-uncased"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded successfully for model {model_name}.")
        return tokenizer
    except Exception as e:
        logger.error(f"An error occurred while loading the tokenizer: {e}")
        raise

def tokenize_function(examples, tokenizer, max_len=512):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)

def prepare_dataset(X, y, tokenizer):
    try:
        if not isinstance(y, list):
            y = y.tolist()
        y = [int(label) for label in y]
        if not isinstance(X, list):
            X = X.tolist()

        dataset = Dataset.from_dict({"text": X, "label": y})
        tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        logger.info("Dataset prepared and tokenized successfully.")
        return tokenized_dataset
    except Exception as e:
        logger.error(f"An error occurred while preparing the dataset: {e}")
        raise

def get_lora_model(model_name="distilbert-base-uncased"):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=["q_lin", "v_lin"] if "distilbert" in model_name else ["query", "value"]
        )
        lora_model = get_peft_model(model, config)
        logger.info(f"LoRA model configured successfully for model {model_name}.")
        return lora_model
    except Exception as e:
        logger.error(f"An error occurred while configuring the LoRA model: {e}")
        raise

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

def train_model(model, tokenized_dataset, output_dir="./results"):
    try:
        train_val_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val_split["train"]
        eval_dataset = train_val_split["test"]

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        logger.info("Model training completed successfully.")
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"Evaluation results: {eval_results}")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        raise

def apply_fine_tuning(df, text_column="combined_text", label_column="label_encoded"):
    try:
        tokenizer = get_tokenizer()
        tokenized_dataset = prepare_dataset(df[text_column], df[label_column], tokenizer)
        lora_model = get_lora_model()
        train_model(lora_model, tokenized_dataset)
        logger.info("Fine-tuning applied successfully.")
    except Exception as e:
        logger.error(f"An error occurred during fine-tuning: {e}")
        raise

