import os
import uuid
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
# We use the free Inference API URL for OpenChat
API_URL = "https://api-inference.huggingface.co/models/openchat/openchat_3.5"
HF_TOKEN = os.getenv("HF_API_TOKEN")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DATA & PROMPTS ---
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

CRISIS_RESOURCES = (
    "\n\n🚨 **Immediate Support Resources:**\n"
    "• National Suicide Prevention Lifeline: 988 (US)\n"
    "• Crisis Text Line: Text HOME to 741741\n"
    "• International Help: https://www.iasp.info/resources/Crisis_Centres/\n"
    "• Emergency Services: 911 or your local emergency number\n"
)

# --- MODELS ---
class ChatRequest(BaseModel):
    messages: list
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# Session storage (In-memory)
session_states = {}

# --- HELPER FUNCTIONS (API BASED) ---

def query_huggingface(payload):
    """Sends a request to the Hugging Face Inference API."""
    if not HF_TOKEN:
        logger.error("HF_API_TOKEN not set in environment variables.")
        raise HTTPException(500, "Server misconfiguration: API Token missing.")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    # If the model is "cold" (loading), HF returns a 503. We might want to wait, 
    # but for now, we'll return an error or a polite loading message.
    if response.status_code == 503:
        raise HTTPException(503, "Model is currently loading on the server. Please try again in 20 seconds.")
    
    if response.status_code != 200:
        logger.error(f"API Error: {response.text}")
        raise HTTPException(response.status_code, f"AI Service Error: {response.text}")
        
    return response.json()

def generate_chat_response_api(messages):
    """Generates a chat response using the API."""
    # Format prompt for OpenChat (simplistic version)
    # OpenChat expects GPT4 Correct User: ... GPT4 Correct Assistant: ...
    prompt_str = ""
    for msg in messages:
        role = "GPT4 Correct User" if msg['role'] == 'user' else "GPT4 Correct Assistant"
        prompt_str += f"{role}: {msg['content']} <|end_of_turn|> "
    
    prompt_str += "GPT4 Correct Assistant:"

    payload = {
        "inputs": prompt_str,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    result = query_huggingface(payload)
    
    # HF returns a list like [{'generated_text': '...'}]
    if isinstance(result, list) and len(result) > 0:
        return result[0].get('generated_text', '').strip()
    return "I'm having trouble thinking right now."

def score_phq9_answer_api(question, answer):
    """
    Uses the API to score the answer 0-3. 
    We force the model to output just a number.
    """
    prompt = (
        f"Instruction: You are a mental health assistant. Analyze the answer to the PHQ-9 question.\n"
        f"Question: {question}\n"
        f"Patient Answer: {answer}\n"
        f"Task: Rate the severity on a scale of 0 to 3, where 0 is 'Not at all', 1 is 'Several days', "
        f"2 is 'More than half the days', and 3 is 'Nearly every day'.\n"
        f"Output ONLY the single digit (0, 1, 2, or 3). Do not explain.\n"
        f"Score:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 2, 
            "temperature": 0.1, # Low temp for deterministic logic
            "return_full_text": False
        }
    }
    
    try:
        result = query_huggingface(payload)
        text_output = result[0].get('generated_text', '').strip()
        # Extract the first digit found
        for char in text_output:
            if char.isdigit():
                score = int(char)
                # Clamp between 0 and 3
                return max(0, min(3, score))
        return 1 # Default fallback if no number found
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return 1 # Default fallback

# --- FASTAPI APP ---
app = FastAPI(title="HopeGuide AI Service (Serverless)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "mode": "Serverless/API",
        "sessions_active": len(session_states)
    }

@app.post("/phq9-chat")
def phq9_chat(req: ChatRequest):
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
            score = score_phq9_answer_api(
                PHQ9_QUESTIONS[state["current_question"] - 1],
                latest_user_msg
            )
            state["phq9_scores"][state["current_question"] - 1] = score
            
            # Suicide question check (Question 9)
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
    ai_reply = generate_chat_response_api(req.messages)
    return {"response": ai_reply, "session_id": session_id}

# IMPORTANT:
# When running on Render, the Start Command should be:
# uvicorn app:app --host 0.0.0.0 --port 10000