from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
import torch
from pathlib import Path
import logging
import os
import mlflow
import mlflow.pyfunc


from openchat_utils import load_model_and_tokenizer, generate_chat_response, score_phq9_answer

# Model name (allow override via env)
MODEL_NAME = os.getenv("MODEL_NAME", "openchat/openchat_3.5")

# --- Fallback local LoRA directory (exists in your repo root) ---
FALLBACK_LORA_DIR = "openchat-phq9-lora"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DATASET_PATH = BASE_DIR / "dataset.jsonl"

# Globals
model = None
tokenizer = None
embedder = None
session_states = {}

# PHQ-9 questions
PHQ9_QUESTIONS = [
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

# Crisis resources text
CRISIS_RESOURCES = (
    "\n\n🚨 **Immediate Support Resources:**\n"
    "• National Suicide Prevention Lifeline: 988 (US)\n"
    "• Crisis Text Line: Text HOME to 741741\n"
    "• International Help: https://www.iasp.info/resources/Crisis_Centres/\n"
    "• Emergency Services: 911 or your local emergency number\n"
)

# Request model
class ChatRequest(BaseModel):
    messages: list
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# Get Production model from MLflow (fallback to local LoRA)
def get_production_model_path() -> Path:
    logger.info(f"Fetching latest Production model for {MODEL_NAME} from MLflow...")
    try:
        from mlflow import artifacts  # use the artifacts submodule explicitly

        model_uri = f"models:/{MODEL_NAME}/Production"
        local_path = mlflow.pyfunc.download_artifacts(model_uri=model_uri)

        logger.info(f"Downloaded Production model to {local_path}")
        return Path(local_path)
    except Exception as e:
        logger.error(f"Failed to fetch Production model from MLflow: {str(e)}")
        logger.warning(f"Falling back to local checkpoint: {FALLBACK_LORA_DIR}")
        return (BASE_DIR / FALLBACK_LORA_DIR)

# FastAPI app
app = FastAPI(title="HopeGuide AI Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.on_event("startup")
def startup_event():
    global model, tokenizer, embedder
    lora_dir = BASE_DIR / "openchat-phq9-lora/checkpoint-1880"
    model, tokenizer, embedder = load_model_and_tokenizer(lora_dir, BASE_DIR / "cache")

    logger.info("AI model loaded and ready")

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "sessions_active": len(session_states)
    }

@app.post("/phq9-chat")
def phq9_chat(req: ChatRequest):
    if not model or not tokenizer:
        raise HTTPException(503, "Model not loaded")

    session_id = req.session_id
    state = session_states.setdefault(session_id, {
        "phq9_scores": [None] * 9,
        "current_question": 0,
        "suicide_risk": False,
        "assessment_complete": False
    })

    latest_user_msg = req.messages[-1]['content'] if req.messages else ""

    # Crisis keyword detection
    crisis_keywords = ["suicide", "kill myself", "end it all", "self-harm", "not want to live"]
    if any(kw in latest_user_msg.lower() for kw in crisis_keywords):
        state["suicide_risk"] = True
        return {
            "response": "I'm deeply concerned about what you're sharing. Your life matters." + CRISIS_RESOURCES,
            "session_id": session_id,
            "crisis_detected": True
        }

    # PHQ-9 assessment logic
    if not state["assessment_complete"]:
        # Score the previous answer (if any)
        if state["current_question"] > 0 and state["phq9_scores"][state["current_question"] - 1] is None:
            score = score_phq9_answer(
                PHQ9_QUESTIONS[state["current_question"] - 1],
                latest_user_msg,
                model,
                tokenizer
            )
            state["phq9_scores"][state["current_question"] - 1] = score
            if state["current_question"] == 9 and score > 0:
                state["suicide_risk"] = True

        # Completed?
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
            result = f"Assessment complete! Your PHQ-9 score is {total_score} ({severity})."
            if total_score >= 10 or state["suicide_risk"]:
                result += CRISIS_RESOURCES
            return {"response": result, "session_id": session_id}

        # Ask next question
        question = PHQ9_QUESTIONS[state["current_question"]]
        state["current_question"] += 1
        return {"response": f"Question {state['current_question']}/9: {question}", "session_id": session_id}

    # Post-assessment free chat
    ai_reply = generate_chat_response(req.messages, model, tokenizer)
    return {"response": ai_reply, "session_id": session_id}
