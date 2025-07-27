from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import torch
import re
import json
import logging
import os
import uuid
from pathlib import Path
import traceback
import warnings
import psutil
# from optimum.intel import INCModelForCausalLM
from transformers import AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

# Set environment variables
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TORCH_USE_RTLD_GLOBAL"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Request schema
class ChatRequest(BaseModel):
    messages: list
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

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

# Use absolute paths
current_dir = Path(__file__).parent.resolve()
lora_dir = current_dir / "openchat-phq9-lora" / "checkpoint-1880"
dataset_path = current_dir / "dataset.jsonl"

# Initialize globals
model = None
tokenizer = None
embedder = None
dataset = []
dataset_embeddings = None

# Session state management
session_states = {}

# Crisis resources
CRISIS_RESOURCES = (
    "\n\n🚨 **Immediate Support Resources:**\n"
    "• National Suicide Prevention Lifeline: 988 (US)\n"
    "• Crisis Text Line: Text HOME to 741741\n"
    "• International Help: https://www.iasp.info/resources/Crisis_Centres/\n"
    "• Emergency Services: 911 or your local emergency number\n\n"
    "You are not alone - help is available right now."
)

# Check library versions for compatibility
def check_library_versions():
    try:
        import torch
        import transformers
        import sentence_transformers
        import peft
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"Sentence Transformers version: {sentence_transformers.__version__}")
        logger.info(f"PEFT version: {peft.__version__}")
    except Exception as e:
        logger.error(f"Error checking library versions: {str(e)}")

# Validate LoRA model directory
def validate_lora_dir(lora_dir: Path):
    logger.info(f"Validating LoRA directory: {lora_dir}")
    if not lora_dir.exists() or not lora_dir.is_dir():
        logger.error(f"LoRA directory not found: {lora_dir}")
        return False

    required_files = ["adapter_config.json"]
    model_file = lora_dir / "adapter_model.safetensors"
    legacy_model_file = lora_dir / "adapter_model.bin"

    if not model_file.exists() and not legacy_model_file.exists():
        logger.error(f"Model file (adapter_model.safetensors or adapter_model.bin) not found in {lora_dir}")
        return False

    if not (lora_dir / "adapter_config.json").exists():
        logger.error(f"adapter_config.json not found in {lora_dir}")
        return False

    # Lightweight check for safetensors file integrity
    if model_file.exists():
        try:
            with open(model_file, "rb") as f:
                header = f.read(8)
                if len(header) < 8:
                    logger.error(f"Invalid safetensors file: {model_file} (incomplete header)")
                    return False
            logger.info(f"LoRA model file {model_file} appears valid")
        except Exception as e:
            logger.error(f"Error validating safetensors file {model_file}: {str(e)}")
            return False

    return True

# Load models
def load_models():
    global model, tokenizer, embedder
    base_model_name = "openchat/openchat_3.5"
    logger.info(f"Preparing to load base model: {base_model_name}")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            padding_side="left",
            trust_remote_code=True,
            cache_dir=current_dir / "cache",
        )
        logger.info("Tokenizer loaded successfully")

        # Safety check for pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",  # Automatically maps to GPU if available
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=current_dir / "cache",
            use_safetensors=True,
        )
        logger.info("Base model loaded successfully")

        # Verify device placement
        logger.info(f"Model device: {next(base_model.parameters()).device}")

        # Load PEFT model
        model = PeftModel.from_pretrained(
            base_model,
            str(lora_dir),
            torch_dtype=torch.float16,
            is_trainable=False,
            device_map="auto",
        )
        model.eval()
        logger.info("PEFT model loaded successfully")

        # Load sentence transformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Sentence transformer loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load and preprocess dataset
def load_dataset():
    global dataset, dataset_embeddings
    if not dataset_path.exists():
        logger.warning(f"Dataset file not found at {dataset_path}")
        logger.warning("Running with fallback response system")
        return False

    try:
        with open(dataset_path, "r") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        
        # Precompute embeddings
        if embedder:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Computing dataset embeddings on {device}")
            dataset_embeddings = torch.stack([
                embedder.encode(entry["messages"][0]["content"], convert_to_tensor=True)
                for entry in dataset
            ])
        
        logger.info(f"Dataset loaded with {len(dataset)} entries")
        return True
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# LLM-based scoring function
def score_response_with_llm(question: str, response: str) -> int:
    """Use the LLM to score responses based on meaning and context"""
    if not response.strip():
        logger.warning("Empty response received, defaulting to score 0")
        return 0

    scoring_prompt = f"""
    [SYSTEM]
    You are a psychiatric assessment specialist scoring PHQ-9 responses.
    Score the following response to the question: {question}
    
    Response: "{response}"
    
    Scoring criteria:
    0 = Not at all
    1 = Several days (less than half the days)
    2 = More than half the days
    3 = Nearly every day
    
    Consider the meaning, context, and emotional content of the response.
    Output ONLY the numerical score (0-3).
    [/SYSTEM]
    [ASSISTANT]
    Score:
    """
    
    inputs = tokenizer(scoring_prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            temperature=0.1,
            pad_token_id=pad_token_id
        )
    
    # Extract and parse score
    score_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    try:
        return int(score_text.strip())
    except ValueError:
        logger.warning(f"LLM returned invalid score: '{score_text}' for response: '{response}'")
        # Fallback scoring for invalid responses
        response_lower = response.lower()
        if "no" in response_lower or "not" in response_lower:
            return 0
        if "sometimes" in response_lower or "occasionally" in response_lower:
            return 1
        if "often" in response_lower or "frequently" in response_lower:
            return 2
        if "every day" in response_lower or "always" in response_lower:
            return 3
        return 1  # Default to mild

# Initialize application
try:
    check_library_versions()
    if not validate_lora_dir(lora_dir):
        logger.warning("LoRA validation failed, skipping model loading")
        model_loaded = False
    else:
        model_loaded = load_models()
    dataset_loaded = load_dataset()
    if not model_loaded or not dataset_loaded:
        logger.warning("API will run in fallback mode")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    logger.error(traceback.format_exc())
    logger.warning("API will run in fallback mode")

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_ready": model is not None,
        "sessions_active": len(session_states),
        "memory_usage": f"{psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB"
    }

@app.get("/")
def read_root():
    return {
        "status": "PHQ-9 Chatbot API is running",
        "model_loaded": model is not None and tokenizer is not None,
        "dataset_loaded": len(dataset) > 0,
        "sessions_active": len(session_states)
    }

@app.post("/phq9-chat")
async def chat(req: ChatRequest):
    try:
        session_id = req.session_id
        
        # Initialize session state
        if session_id not in session_states:
            session_states[session_id] = {
                "phq9_scores": [None] * 9,
                "current_question": 0,
                "suicide_risk": False,
                "assessment_complete": False,
                "last_user_response": None
            }
        state = session_states[session_id]
        
        # Fallback response if models aren't loaded
        if model is None or tokenizer is None:
            logger.warning("No model or tokenizer loaded, using fallback response")
            return {
                "response": "I'm here to listen and support you. While I'm currently running in a limited mode, I want you to know that your feelings are valid and important. Could you tell me more about what's been on your mind lately?",
                "session_id": session_id
            }

        # Extract latest user message
        latest_user_message = req.messages[-1]['content'] if req.messages else ""
        
        # Check for crisis keywords
        crisis_keywords = ["suicide", "end it all", "don't want to live", "kill myself", "self-harm"]
        if any(keyword in latest_user_message.lower() for keyword in crisis_keywords):
            state["suicide_risk"] = True
            return {
                "response": "I'm deeply concerned about what you're sharing. Your life matters immensely." + CRISIS_RESOURCES,
                "session_id": session_id,
                "crisis_detected": True
            }

        # PHQ-9 assessment flow - Simple logic
        if not state["assessment_complete"]:
    # Check if waiting for answer to a previous question
            if state["current_question"] > 0 and state["phq9_scores"][state["current_question"] - 1] is None:
                if latest_user_message and latest_user_message.strip():
                    prev_q_idx = state["current_question"] - 1
                    question_text = phq9_questions[prev_q_idx]
                    score = score_response_with_llm(question_text, latest_user_message)
                    state["phq9_scores"][prev_q_idx] = score
                    logger.info(f"Scored Q{prev_q_idx+1}: '{latest_user_message}' => {score}")

            # Suicide risk check
                    if prev_q_idx == 8 and score > 0:
                        state["suicide_risk"] = True

                else:
            # User hasn't answered yet
                    return {
                "response": f"(Waiting for your answer to Question {state['current_question']}/9: {phq9_questions[state['current_question'] - 1]})",
                "session_id": session_id,
                "current_question": state["current_question"] - 1,
                "question_count": 9
            }

    # Check if PHQ-9 is complete
            if state["current_question"] >= 9:
                state["assessment_complete"] = True
                total_score = sum(s for s in state["phq9_scores"] if s is not None)
                severity = (
            "minimal" if total_score <= 4 else
            "mild" if total_score <= 9 else
            "moderate" if total_score <= 14 else
            "moderately severe" if total_score <= 19 else
            "severe"
        )
                assessment = (
            f"Assessment complete! Your PHQ-9 score is {total_score}, suggesting {severity} depression. "
            "This is a screening tool, not a diagnosis. Please consult a healthcare professional."
        )
                if total_score >= 10 or state["suicide_risk"]:
                    assessment += CRISIS_RESOURCES

                return {
            "response": assessment,
            "session_id": session_id,
            "assessment_complete": True,
            "phq9_score": total_score
        }

    # Ask the next question
            question = phq9_questions[state["current_question"]]
            question_number = state["current_question"] + 1
            state["current_question"] += 1

            return {
        "response": f"Question {question_number}/9: {question}",
        "session_id": session_id,
        "current_question": state["current_question"] - 1,
        "question_count": 9
    }

        # Post-assessment conversation using model
        system_prompt = (
            "You are HopeGuide, an empathetic psychiatrist AI. "
            "The PHQ-9 assessment is complete. Now, focus on:\n"
            "1. Validating the user's feelings\n"
            "2. Offering CBT-inspired strategies\n"
            "3. Providing hope and encouragement\n"
            "4. Monitoring for crisis keywords and providing resources if needed\n"
            "Respond empathetically and conversationally."
        )

        # Build conversation
        conversation_history = [
            {"role": "system", "content": system_prompt},
            *req.messages
        ]

        # Apply chat template
        try:
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history]) + "\nAssistant:"
        except Exception as e:
            logger.error(f"Error applying chat template: {str(e)}")
            return {
                "response": "I'm having trouble processing your request right now, but I'm here to help. Could you share a bit more about how you're feeling?",
                "session_id": session_id
            }

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Pad token fallback
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=1.2,
                pad_token_id=pad_token_id
            )

        new_tokens = outputs[0, input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean and format response
        response = re.sub(r'<.*?>|\[.*?\]|[^\x00-\x7F]+', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()
        sentences = re.split(r'(?<=[.!?]) +', response)
        sentences = [s.capitalize().strip() for s in sentences if s.strip()]
        final_response = ' '.join(sentences)

        return {
            "response": final_response or "I'm here to listen. Could you tell me more about how you're feeling?",
            "session_id": session_id
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "response": "I'm experiencing some technical difficulties right now, but I want you to know that I'm here for you. Your mental health matters, and it's important that you reach out for support. Could you tell me what's been weighing on your mind?",
            "session_id": session_id
        }