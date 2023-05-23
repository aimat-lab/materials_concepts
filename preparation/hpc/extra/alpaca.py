from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
)

from peft import PeftModel

tokenizer = LlamaTokenizerFast.from_pretrained(
    "./llama-7B/", return_tensors="pt", paddding_side="left"
)
tokenizer.pad_token_id = 0

model = LlamaForCausalLM.from_pretrained(
    "./llama-7B/", load_in_8bit=True, device_map="auto"
)

model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b")

model.eval()

batch = [
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500.",
]

inputs = tokenizer(
    batch,
    padding="longest",
    return_tensors="pt",
    add_special_tokens=False,
).to("cuda")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=512,
    temperature=0,
)

generated_ids = [output.detach().cpu().numpy() for output in outputs]
generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

for i, text in enumerate(generated_texts):
    print(f"Input: {batch[i]}")
    print(f"Output: {text}\n")
    print("=" * 80)
