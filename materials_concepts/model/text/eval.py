import logging
import os
import pickle
import sys

import fire
import pandas as pd
import torch
from datasets import Dataset as HfDataset
from datasets import load_from_disk
from materials_concepts.model.metrics import print_metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from materials_concepts.utils.utils import (
    save_compressed,
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


def main(
    data_path: str = "data-v2/model/val.data.M.pkl",
    lookup_path: str = "data-v2/table/lookup/lookup.M.csv",
    processed_data_dir: str = "data-v2/text-baseline/processed_data_test",
    fine_tuned_model: str = "data-v2/text-baseline/final_model",
    eval_batch_size: int = 1024,
    max_length: int = 32,
    output_dir: str = "data-v2/text-baseline/results",
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data and lookup map
    data = load_data(data_path)
    id_to_concept = create_id_to_concept_map(lookup_path)

    # Load tokenizer and model from fine-tuned path
    logger.info(f"Loading fine-tuned model from {fine_tuned_model}")
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        fine_tuned_model, num_labels=2
    )

    # Prepare datasets
    test_dataset_path = os.path.join(processed_data_dir, "test")

    if os.path.exists(test_dataset_path):
        logger.info(f"Loading processed datasets from {processed_data_dir}...")
        test_dataset = load_from_disk(test_dataset_path)
    else:
        logger.info("Processed datasets not found. Creating and saving them...")
        test_dataset = prepare_dataset(
            data["X_test"], data["y_test"], id_to_concept, tokenizer, max_length
        )

        logger.info(f"Saving processed datasets to {processed_data_dir}...")
        test_dataset.save_to_disk(test_dataset_path)

    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Set up Trainer for evaluation
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=eval_batch_size,
        do_train=False,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # Evaluate on the test set
    logger.info("Evaluating model on the test set...")
    predictions = trainer.predict(test_dataset)
    
    scores = torch.tensor(predictions.predictions).softmax(dim=-1)[:, 1].numpy()
    labels = predictions.label_ids

    # Print and save metrics
    metrics_save_path = os.path.join(output_dir, "test_metrics.json")
    logger.info(f"Saving metrics to {metrics_save_path}")
    print_metrics(labels, scores, save_path=metrics_save_path)

    logger.info("Evaluation complete.")

    save_compressed(
        scores,
        os.path.join(output_dir, "test_predictions.pkl.gz"),
    )


if __name__ == "__main__":
    fire.Fire(main)