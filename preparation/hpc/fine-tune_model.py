from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import logging
from importlib import reload

reload(logging)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

print("Loading model...")
tokenizer = LlamaTokenizer.from_pretrained("./HF/")
print("Loading tokenizer...")
model = LlamaForCausalLM.from_pretrained("./HF/")

CUTOFF_LENGTH = 2048


class ConceptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row["abstract"]
        tags = str(row["tags"].split(","))

        text_ids = tokenizer(
            text, padding=True, max_length=CUTOFF_LENGTH, return_tensors="pt"
        ).input_ids
        tags_ids = tokenizer(
            tags, padding=True, max_length=CUTOFF_LENGTH, return_tensors="pt"
        ).input_ids
        return {"inputs": text_ids, "labels": tags_ids}


print("Loading dataset...")
train_dataset = ConceptDataset(pd.read_csv("train.csv"))


# define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Adjust the parameters as needed

# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # your own training dataset
    optimizers=(optimizer, scheduler),
)

print("Training...")
# fine-tune the model
trainer.train()
