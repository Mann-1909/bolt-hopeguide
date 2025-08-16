from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "openchat/openchat_3.5"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model (CPU only, full precision)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,   # use float32 on CPU
    device_map="cpu"              # disable auto device mapping
)

# Basic prompt
prompt = "What is the difference between AI and machine learning?"

# Tokenize and send to CPU
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# Decode response
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
