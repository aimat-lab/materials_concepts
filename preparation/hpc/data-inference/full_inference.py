from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
from peft import PeftModel
import pandas as pd
import fire
import os


#  500 samples == 20 min
# 1500 samples == 60 min
# 4500 samples == 3h
# => 4000 samples


# 3h


def get_model_and_tokenizer(llama_variant, model_id):
    tokenizer = LlamaTokenizerFast.from_pretrained(
        f"./llama-{llama_variant}/", return_tensors="pt", paddding_side="left"
    )
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        f"./llama-{llama_variant}/", load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, f"./models/{llama_variant}/{model_id}")
    model.eval()

    return model, tokenizer


def get_prepared_data(input_file, start, n):
    df = pd.read_csv(input_file).iloc[start : start + n][["id", "abstract"]]
    df.abstract = df.abstract.apply(
        lambda text: "<s>" + text + "\n\n\n###\nKEYWORDS:\n###\n\n\n"
    )
    return df


def create_output_file(output_file):
    print("Creating output file:", output_file)
    header = pd.DataFrame(columns=["id", "concepts"])
    header.to_csv(output_file, index=False)


# 004000_008000.csv


def get_batch(data, i, batch_size):
    return list(data.iloc[i : i + batch_size].abstract)


def get_ids(data, i, batch_size):
    return list(data.iloc[i : i + batch_size].id)


def generate_batch(model, tokenizer, batch, max_new_tokens):
    inputs = tokenizer(
        batch,
        padding="longest",  # Pad the sequences in the batch to the same length
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=0,
    )

    generated_ids = [output.detach().cpu().numpy() for output in outputs]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def extract_concepts(batch_texts):
    CONCEPTS_START = "###\nKEYWORDS:\n###"
    return [text.split(CONCEPTS_START)[1].strip() for text in batch_texts]


def save_batch(ids, concepts, output_file):
    df_batch = pd.DataFrame({"id": ids, "concepts": concepts})
    df_batch.to_csv(output_file, mode="a", header=False, index=False)


def main(
    n=3_000,
    start=0,
    llama_variant="13B",
    model_id="full-finetune",
    input_file="data/works.csv",
    batch_size=25,
    max_new_tokens=512,
):
    settings = {
        "llama_variant": llama_variant,
        "model_id": model_id,
        "start": start,
        "end": start + n,
        "n": n,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
    }
    print("Settings:", settings)

    OUTPUT_PATH = f"./data/inference_{llama_variant}/{model_id}/"
    FILE = f"{start:06}_{start + n:06}.csv"
    OUTPUT_FILE = os.path.join(OUTPUT_PATH, FILE)

    model, tokenizer = get_model_and_tokenizer(llama_variant, model_id)

    data = get_prepared_data(input_file, start, n)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    create_output_file(OUTPUT_FILE)

    for i in range(0, len(data), batch_size):
        batch = get_batch(data, i, batch_size)

        batch_texts = generate_batch(model, tokenizer, batch, max_new_tokens)
        batch_concepts = extract_concepts(batch_texts)

        batch_ids = get_ids(data, i, batch_size)
        save_batch(batch_ids, batch_concepts, OUTPUT_FILE)


if __name__ == "__main__":
    fire.Fire(main)
