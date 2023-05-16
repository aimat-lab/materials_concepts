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

text = """
Isotropic Channel Mobility inon 4HSiCCFace with Vicinal Off-Angle. UMOSFET is
theoretically suitable to decrease the on-resistance of the MOSFET. In this
study, in order to determine the cell structure of the SiCUMOSFET with extremely
low on-resistance, influences of the orientation of the trench and the off-angle
of the wafer on the MOS properties are investigated. The channel resistance,
gate IV curves and instability of threshold voltage are superior on the {11-20}
planes as compared with other planes. On the vicinal off wafer, influence of the
off-angle disappears and the properties on the equivalent planes are almost the
same. The obtained results indicate that the extremely low on-resistance with
the high stability and high reliability is possible in the SiCUMOSFET by the
hexagonal cell composed of the six {11-20} planes on the vicinal off wafer, and
actually an extremely low channel resistance was demonstrated on the hexagonal
UMOSFET with the six {11-20} planes on the vicinal off wafer.
"""

text = text + " ==> "
text = text.replace("\n", " ")

with torch.no_grad():
    outputs = model.generate(
        input_ids=tokenizer(text, return_tensors="pt")["input_ids"].to("cuda"),
        max_new_tokens=512,
    )
    print(
        tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0]
    )
