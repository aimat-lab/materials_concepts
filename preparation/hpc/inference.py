from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)
from peft import PeftModel
import pandas as pd
import time

USE_FINE_TUNED_MODEL = True

MODEL_ID = "./models/working/"

model = LlamaForCausalLM.from_pretrained(
    "./llama-7B/", load_in_8bit=True, device_map="auto"
)
tokenizer = LlamaTokenizer.from_pretrained("./llama-7B/", return_tensors="pt")
tokenizer.pad_token_id = 0
tokenizer.paddding_side = "left"

if USE_FINE_TUNED_MODEL:
    model = PeftModel.from_pretrained(model, MODEL_ID)

model.eval()

df = pd.read_csv("./data/inference.csv").head(10)
df.abstract = df.abstract.apply(lambda text: text + " ==> ")  # prepare


batch_size = 5
abstracts = list(df.abstract)

start_time = time.time()
for i in range(0, len(abstracts), batch_size):
    batch = abstracts[i : i + batch_size]

    # Pad the sequences in the batch to the same length
    inputs = tokenizer(batch, padding="longest", return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=512,
        temperature=0,
    )

    generated_ids = [output.detach().cpu().numpy() for output in outputs]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # assign batch to df
    # df.loc[i : i + batch_size - 1, "generated"] = generated_texts

    for index, text in enumerate(generated_texts):
        id = df.loc[i + index, "id"]
        print(id)
        print(text)
        print("=====================================")

    print("Batch processing took: ", time.time() - start_time, "\n")
    start_time = time.time()
