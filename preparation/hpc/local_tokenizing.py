from transformers import (
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
)
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split

tokenizer = LlamaTokenizer.from_pretrained("./HF/", return_tensors="pt")
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token


class ConceptDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row["abstract"] + " ==> " + str(row["tags"].split(","))

        text_encodings = tokenizer(
            text,
            return_tensors="pt",
        )

        return {
            "input_ids": text_encodings["input_ids"].flatten(),
            "attention_mask": text_encodings["attention_mask"].flatten(),
        }


print("Loading dataset...")
dataset = ConceptDataset(pd.read_csv("train.csv"))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # Let's say we want 80% of the data for training
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=data_collator
)
