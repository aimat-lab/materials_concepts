from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
from peft import PeftModel
import torch
import pandas as pd

USE_FINE_TUNED_MODEL = True

MODEL_ID = "./results/"

model = LlamaForCausalLM.from_pretrained("./HF/", load_in_8bit=True, device_map="auto")
tokenizer = LlamaTokenizer.from_pretrained("./HF/", return_tensors="pt")
if USE_FINE_TUNED_MODEL:
    model = PeftModel.from_pretrained(model, MODEL_ID)

model.eval()

df = pd.read_csv("train.csv")
df.abstract = df.abstract.apply(lambda text: text + " ==> ")  # prepare


with torch.no_grad():
    for abstract in list(df.abstract):
        outputs = model.generate(
            input_ids=tokenizer(abstract, return_tensors="pt")["input_ids"].to("cuda"),
            max_new_tokens=200,
        )
        print(
            tokenizer.batch_decode(
                outputs.detach().cpu().numpy(), skip_special_tokens=True
            )[0]
        )
