from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "openchat/openchat_3.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True,
)

# Basic prompt
prompt = "What is the difference between AI and machine learning?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
