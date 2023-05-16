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
    get_peft_model_state_dict,
)

from torch.utils.data import Dataset
import pandas as pd
import logging
from importlib import reload

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reload(logging)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("./HF/", return_tensors="pt")
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

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
            return_tensors="pt",
        )

        return {
            "input_ids": text_encodings["input_ids"].flatten().to(device),
            "attention_mask": text_encodings["attention_mask"].flatten().to(device),
        }


print("Loading dataset...")
train_dataset = ConceptDataset(pd.read_csv("train.csv"))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


## Setup LORA

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # attention heads
    lora_alpha=16,  # alpha scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    inference_mode=False,
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
model = prepare_model_for_int8_training(model)

# define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=10,
    weight_decay=0.01,
    per_device_train_batch_size=16,  # reduce the batch size
    gradient_accumulation_steps=1,  # add gradient accumulation
    logging_dir="./logs",
    logging_steps=10,
    optim="adamw_torch",
    save_strategy="epoch",
    learning_rate=1e-4,
)


# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # your own training dataset
    data_collator=data_collator,
)


## PEFT
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

model = torch.compile(model)

print("Training...")
# fine-tune the model
trainer.train()

model.save_pretrained("./results")
