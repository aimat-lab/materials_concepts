from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
)
import torch
import pandas as pd
import numpy as np
from importlib import reload
import random
import logging
import json
import fire


reload(logging)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## Deterministic training
seed = 42
set_seed(seed)


def main(
    llama_variant="7B",
    output_dir="finetuned",
    size_train_dataset=0.9,
    tokenizer_max_length=1024,
    num_epochs=2,
    weight_decay=0.005,
    batch_size=1,
    lr=1e-3,
):
    ## Constants
    DATA_PATH = "./data/train.csv"
    MODEL_PATH = f"./llama-{llama_variant}/"
    OUTPUT_MODEL_PATH = f"./models/{llama_variant}/{output_dir}/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Setup
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(
        MODEL_PATH, return_tensors="pt", padding_side="left"
    )
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    class ConceptDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]

            text = (
                "<s>"
                + row["abstract"]
                + "\n\n\n###\nKEYWORDS:\n###\n\n\n"
                + str(row["tags"].split(","))
                + "</s>"
            )

            # Tokenize text and tags separately
            text_encodings = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer_max_length,
            )

            return {
                "input_ids": text_encodings["input_ids"].flatten().to(device),
                "attention_mask": text_encodings["attention_mask"].flatten().to(device),
                "labels": text_encodings["input_ids"].flatten().to(device),
            }

    ## Dataset
    print("Loading dataset...")
    dataset = ConceptDataset(pd.read_csv(DATA_PATH))

    dataset_size = len(dataset)
    train_size = int(size_train_dataset * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    ## Setup LORA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # attention heads
        lora_alpha=32,  # alpha scaling
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
        inference_mode=False,
    )

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    ## Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        logging_dir="./logs",
        logging_steps=1,
        logging_strategy="epoch",
        optim="adamw_torch",
        learning_rate=lr,
        evaluation_strategy="epoch" if size_train_dataset < 1 else "no",
    )

    ## Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Training...")
    model.train()
    trainer.train()

    model.save_pretrained(OUTPUT_MODEL_PATH)

    settings = {
        "size_train_dataset": size_train_dataset,
        "tokenizer_max_length": tokenizer_max_length,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "lr": lr,
    }

    with open(f"{OUTPUT_MODEL_PATH}/settings.json", "w") as outfile:
        json.dump(settings, outfile)


if __name__ == "__main__":
    fire.Fire(main)
