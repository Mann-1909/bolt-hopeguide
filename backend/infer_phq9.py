from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json
from sentence_transformers import SentenceTransformer, util
import torch
import re
import logging

# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

base_model_name = "openchat/openchat_3.5"
lora_dir = "/home/b221265ec/Bolt/OpenChat/openchat-phq9-lora/checkpoint-564"

# Configure bitsandbytes properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # Match compute dtype to input
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # Consistent dtype
    quantization_config=bnb_config
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
    f"As HopeGuide, an empathetic psychiatrist, support someone with PHQ-9 score {{best_score}} by choosing a single, fitting metaphor (e.g., tree for resilience, light for hope, garden for nurturing, path for journey) based on their emotional tone and context. "
    "Follow this structure: 1) Validate their feelings, e.g., 'I hear your pain about [specific mention]...'; "
    "2) Identify a strength, e.g., 'Your [resilience/courage] shines like...'; "
    "3) Suggest a unique CBT-inspired coping strategy (e.g., reframing negative thoughts, self-compassion, or 5-4-3-2-1 grounding) without repeating ideas; "
    "4) Envision a hopeful future, e.g., 'Imagine a day when...'; "
    "5) End with an open question, e.g., 'What feels possible today?' "
    "Use warm, poetic language in 2-3 sentences, tailoring responses to their context (e.g., loss of parents) and avoiding reliance on example stories. "
    "Ensure responses are concise, evidence-based, and hopeful. For scores >20, gently suggest professional resources like the National Suicide Prevention Lifeline."
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
    
    # Generate response - use sampling with temperature
    with torch.no_grad():
        outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,  # Lower for coherence
    repetition_penalty=1.2,  # Less aggressive
    pad_token_id=tokenizer.eos_token_id
)
    
    # Decode only the new tokens
    new_tokens = outputs[0, input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up response
    response = response.split("User:")[0]
    response = re.sub(r'^(System|User|Assistant):?\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'<.*?>', '', response)  # Remove HTML tags
    response = re.sub(r'\[.*?\]', '', response)  # Remove markdown links
    response = re.sub(r'[^\x00-\x7F]+', ' ', response)  # Remove non-ASCII
    response = re.sub(r'\s+', ' ', response).strip()  # Normalize whitespace
    # Sentence case and paragraph formatting
    sentences = re.split(r'(?<=[.!?]) +', response)
    sentences = [s.capitalize().strip() for s in sentences if s.strip()]
    response = '\n'.join(sentences)
    # Handle empty responses
    if not response:
        response = "I'm here to listen. Could you tell me more about how you're feeling?"
    
    # Avoid repetition
    if response in previous_responses:
        response = "I appreciate you sharing that. Could you tell me more about what's been on your mind lately?"
    else:
        previous_responses.add(response)
    
    # Print only the bot's natural response
    print(f"\nBot: {response}\n")
    
    # Add to conversation history
    conversation_history.append({"role": "assistant", "content": response})
    
    # Maintain conversation history
    if len(conversation_history) > 7:
        conversation_history = [conversation_history[0]] + conversation_history[-6:]