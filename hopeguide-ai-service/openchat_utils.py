from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import torch
import re
import mlflow
from pathlib import Path
import logging

# ===== MLflow Config =====
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "openchat/openchat_3.5"
FALLBACK_LORA_DIR = Path(__file__).parent / "openchat-phq9-lora" / "checkpoint-1880"
# =========================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_production_model_path():
    """Fetch the latest Production model from MLflow or fall back to local checkpoint."""
    logger.info(f"Fetching latest Production model for {MODEL_NAME} from MLflow...")
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        local_path = mlflow.artifacts.download_artifacts(model_uri=model_uri)
        logger.info(f"✅ Downloaded Production model to {local_path}")
        return Path(local_path)
    except Exception as e:
        logger.error(f"❌ Failed to fetch Production model from MLflow: {str(e)}")
        logger.warning(f"Using fallback model at {FALLBACK_LORA_DIR}")
        return FALLBACK_LORA_DIR

def load_model_and_tokenizer(lora_dir=None, cache_dir=None):
    """
    Load OpenChat model with LoRA weights in full precision (CPU-only).
    """
    if lora_dir is None:
        lora_dir = get_production_model_path()

    base_model_name = "openchat/openchat_3.5"
    device = "cpu"  # Force CPU

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        padding_side="left", 
        cache_dir=cache_dir
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model in float32 on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map={"": device},
        cache_dir=cache_dir
    )

    # Load LoRA weights on CPU
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        torch_dtype=torch.float32,
        device_map={"": device}
    )
    model.eval()

    # Sentence embeddings for PHQ-9 or other use cases
    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2", 
        device=device
    )

    return model, tokenizer, embedder

def score_phq9_answer(question, answer, model, tokenizer):
    prompt = f"""
    [SYSTEM]
    You are a psychiatric assessment specialist scoring PHQ-9 responses.
    Question: {question}
    Response: "{answer}"
    Output ONLY a score: 0, 1, 2, or 3.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False, temperature=0.1)
    try:
        return int(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
    except:
        return 1

def generate_chat_response(messages, model, tokenizer):
    """
    Generate a chat response given a list of messages using CPU-only model.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    response = re.sub(r'\s+', ' ', response).strip()
    return response
