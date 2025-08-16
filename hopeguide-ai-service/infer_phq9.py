from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from sentence_transformers import SentenceTransformer, util
import torch
import re
import logging

# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

base_model_name = "openchat/openchat_3.5"
lora_dir = "/home/b221265ec/Bolt/OpenChat/openchat-phq9-lora/checkpoint-564"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Load base model without bitsandbytes
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="cpu",       # change to "cuda" if GPU is available
    torch_dtype=torch.float16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_dir)
model.eval()

# PHQ-9 questions
phq9_questions = [
    "Over the last two weeks, have you found little interest or pleasure in doing things?",
    "Have you been feeling down, depressed, or hopeless?",
    "Have you had trouble falling or staying asleep, or slept too much?",
    "Have you felt tired or had little energy?",
    "Have you had poor appetite or tended to overeat?",
    "Have you felt bad about yourself, or that you are a failure or have let yourself or your family down?",
    "Have you had trouble concentrating on things, such as reading, work, or watching television?",
    "Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?",
    "Have you had thoughts of self-harm or felt that you would be better off dead?"
]

# Load dataset for similarity
with open("/home/b221265ec/Bolt/OpenChat/dataset.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Collect PHQ-9 answers with safety checks
user_answers = []
print("Let's start with a few questions to understand how you've been feeling. Answer honestly.\n")
for i, q in enumerate(phq9_questions):
    ans = input(f"{q}\nYou: ")
    user_answers.append(ans)
    
    # Immediate crisis response
    if ("die" in ans.lower() or "self-harm" in ans.lower() or 
        ("not want to live" in ans.lower() and i == len(phq9_questions)-1)):
        print("\n🚨 Please reach out for immediate help:")
        print("National Suicide Prevention Lifeline: 988")
        print("Crisis Text Line: Text HOME to 741741")
        print("International: https://www.iasp.info/resources/Crisis_Centres/")
        print("You're not alone - help is available right now.\n")

user_profile = " ".join(user_answers)

# Find most similar dataset entry
max_sim = -1
best_score = "unknown"
for entry in dataset:
    user_msg = entry["messages"][0]["content"]
    emb1 = embedder.encode(user_profile, convert_to_tensor=True)
    emb2 = embedder.encode(user_msg, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb1, emb2).item()
    if sim > max_sim:
        max_sim = sim
        assistant_msg = entry["messages"][1]["content"]
        if "score is" in assistant_msg:
            score_parts = assistant_msg.split("score is ")
            if len(score_parts) > 1:
                best_score = score_parts[1].split()[0].strip('.,')

# Natural system prompt
system_prompt = (
    f"As HopeGuide, an empathetic psychiatrist, support someone with PHQ-9 score {{best_score}} by choosing a single, fitting metaphor..."
)

# Start chat with clean history
conversation_history = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "My PHQ-9 symptoms: " + user_profile},
    {"role": "assistant", "content": f"I understand your PHQ-9 score is {best_score}. How can I support you right now?"}
]

print(f"\nBased on your responses, your PHQ-9 score is {best_score}. How can I help? (Type 'exit' to quit)\n")

# Previous responses tracking
previous_responses = set()

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Take care. Remember support is always available.")
        break
        
    # Emergency keyword detection
    emergency_keywords = ["kill myself", "end it", "want to die", "not want to live", "suicide"]
    if any(kw in user_input.lower() for kw in emergency_keywords):
        print("\n🚨 Please contact emergency services immediately:")
        print("National Suicide Prevention Lifeline: 988")
        print("Crisis Text Line: Text HOME to 741741")
        print("You matter, and help is available right now.\n")
        continue
        
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})
    
    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history]) + "\nAssistant:"
    
    # Tokenize inputs
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean response
    new_tokens = outputs[0, input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = re.sub(r'^(System|User|Assistant):?\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'<.*?>', '', response)
    response = re.sub(r'\[.*?\]', '', response)
    response = re.sub(r'[^\x00-\x7F]+', ' ', response)
    response = re.sub(r'\s+', ' ', response).strip()
    sentences = re.split(r'(?<=[.!?]) +', response)
    sentences = [s.capitalize().strip() for s in sentences if s.strip()]
    response = '\n'.join(sentences)
    
    if not response:
        response = "I'm here to listen. Could you tell me more about how you're feeling?"
    
    if response in previous_responses:
        response = "I appreciate you sharing that. Could you tell me more about what's been on your mind lately?"
    else:
        previous_responses.add(response)
    
    print(f"\nBot: {response}\n")
    conversation_history.append({"role": "assistant", "content": response})
    
    # Maintain conversation history
    if len(conversation_history) > 7:
        conversation_history = [conversation_history[0]] + conversation_history[-6:]
