import logging
import os
import pickle
import sys

import fire
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HfDataset
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> dict:
    """Loads the pickled data file."""
    logger.info(f"Loading data from {data_path}")
    with open(data_path, "rb") as f:
        return pickle.load(f)


def create_id_to_concept_map(lookup_path: str) -> dict:
    """Creates an efficient ID to concept name mapping."""
    logger.info(f"Creating ID-to-concept map from {lookup_path}")
    lookup_df = pd.read_csv(lookup_path)
    return pd.Series(lookup_df.concept.values, index=lookup_df.id).to_dict()


def prepare_dataset(
    X: list[tuple[int, int]],
    y: list[int],
    id_map: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> HfDataset:
    """Prepares a dataset for the transformer model."""

    def tokenize_function(examples):
        # Get concept names from IDs
        concepts1 = [id_map[pair[0]] for pair in examples["pairs"]]
        concepts2 = [id_map[pair[1]] for pair in examples["pairs"]]
        # Tokenize the pairs
        return tokenizer(
            concepts1,
            concepts2,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    # Create a dictionary for the dataset
    data_dict = {"pairs": X, "labels": y}
    dataset = HfDataset.from_dict(data_dict)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["pairs"]
    )
    return tokenized_dataset


def compute_metrics(eval_pred):
    """Computes AUC and accuracy for evaluation."""
    predictions, labels = eval_pred
    # The output of the model is logits, so we apply softmax to get probabilities for multi-class
    # or sigmoid for binary. For binary classification, we can just take the logit of the positive class.
    # Here we assume the positive class is at index 1.
    scores = torch.tensor(predictions).softmax(dim=-1)[:, 1].numpy()
    
    auc = roc_auc_score(labels, scores)
    # For accuracy, we can use argmax on the logits
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    
    return {"auc": auc, "accuracy": acc}


def main(
    data_path: str = "data-v2/model/data.M.pkl",
    lookup_path: str = "data-v2/table/lookup/lookup.M.csv",
    processed_data_dir: str = "data-v2/text-baseline/processed_data",
    model_name: str = "m3rg-iitd/matscibert",
    output_dir: str = "data-v2/text-baseline/",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    max_length: int = 32,
    seed: int = 42,
):
    """
    Fine-tunes a transformer model for link prediction based on concept names.

    Args:
        data_path: Path to the input data file (.pkl).
        lookup_path: Path to the concept ID to name lookup CSV.
        processed_data_dir: Directory to save/load cached processed datasets.
        model_name: Name of the pre-trained model from Hugging Face Hub.
        output_dir: Directory to save the trained model and results.
        num_epochs: Number of training epochs.
        batch_size: Training and evaluation batch size.
        learning_rate: The learning rate for the AdamW optimizer.
        max_length: Maximum sequence length for the tokenizer.
        seed: Random seed for reproducibility.
    """
    logger.info(f"Starting fine-tuning process with seed: {seed}")

    # Load data and lookup map
    data = load_data(data_path)
    id_to_concept = create_id_to_concept_map(lookup_path)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Prepare datasets
    train_dataset_path = os.path.join(processed_data_dir, "train")
    val_dataset_path = os.path.join(processed_data_dir, "val")
    test_dataset_path = os.path.join(processed_data_dir, "test")

    if (
        os.path.exists(train_dataset_path)
        and os.path.exists(val_dataset_path)
        and os.path.exists(test_dataset_path)
    ):
        logger.info(f"Loading processed datasets from {processed_data_dir}...")
        train_dataset = load_from_disk(train_dataset_path)
        val_dataset = load_from_disk(val_dataset_path)
        test_dataset = load_from_disk(test_dataset_path)
    else:
        logger.info("Processed datasets not found. Creating and saving them...")
        train_dataset = prepare_dataset(
            data["X_train"], data["y_train"], id_to_concept, tokenizer, max_length
        )
        val_dataset = prepare_dataset(
            data["X_val"], data["y_val"], id_to_concept, tokenizer, max_length
        )
        test_dataset = prepare_dataset(
            data["X_test"], data["y_test"], id_to_concept, tokenizer, max_length
        )

        logger.info(f"Saving processed datasets to {processed_data_dir}...")
        train_dataset.save_to_disk(train_dataset_path)
        val_dataset.save_to_disk(val_dataset_path)
        test_dataset.save_to_disk(test_dataset_path)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        report_to="tensorboard",
        seed=seed,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train the model
    logger.info("Starting model training...")
    trainer.train()

    # Evaluate on the test set
    logger.info("Evaluating model on the test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    logger.info(f"Test Set Metrics: {test_results}")

    # Save the final model and tokenizer
    trainer.save_model(f"{output_dir}/final_model")
    logger.info(f"Fine-tuning complete. Model saved to {output_dir}/final_model")


if __name__ == "__main__":
    fire.Fire(main)