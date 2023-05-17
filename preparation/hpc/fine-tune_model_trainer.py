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

from torch.utils.data import Dataset, random_split
import pandas as pd
import logging
from importlib import reload

import torch

reload(logging)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("./HF/", return_tensors="pt")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2


print("Loading model...")
model = LlamaForCausalLM.from_pretrained(
    "./HF/",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

## Dataset


class ConceptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row["abstract"] + " ==> " + str(row["tags"].split(","))

        # Tokenize text and tags separately
        text_encodings = tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
        )

        return {
            "input_ids": text_encodings["input_ids"].flatten().to(device),
            "attention_mask": text_encodings["attention_mask"].flatten().to(device),
            "labels": text_encodings["input_ids"].flatten().to(device),
        }


print("Loading dataset...")
dataset = ConceptDataset(pd.read_csv("train.csv"))

dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


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

# define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=2,
    weight_decay=0.005,
    per_device_train_batch_size=1,  # reduce the batch size
    logging_dir="./logs",
    logging_steps=1,
    logging_strategy="steps",
    optim="adamw_torch",
    learning_rate=1e-4,
    evaluation_strategy="epoch",
)


# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # your own training dataset
    eval_dataset=val_dataset,  # your own evaluation dataset
    data_collator=data_collator,
    # compute metrics
)

print("Training...")
# fine-tune the model
model.train()
trainer.train()

model.save_pretrained("./results")
