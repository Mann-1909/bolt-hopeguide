from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time

app = FastAPI()

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173","https://bolt-hopeguide.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    messages: list

# Sample responses for testing
sample_responses = [
    "I hear that you're going through a difficult time. Your courage to reach out shows incredible strength, like a lighthouse standing firm against the storm. Try the 5-4-3-2-1 grounding technique: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. Imagine a day when you feel more at peace with yourself. What small step feels possible for you today?",
    
    "Thank you for sharing your feelings with me. Your willingness to be vulnerable is like a seed that's ready to grow, even in difficult soil. Consider practicing self-compassion by speaking to yourself as you would to a dear friend. Picture a future where you feel more connected to your inner strength. What would bring you a moment of comfort right now?",
    
    "I can sense the weight you're carrying, and I want you to know that your feelings are completely valid. Your resilience reminds me of a tree that bends in the wind but doesn't break. Try writing down three things you're grateful for each day, no matter how small. Envision a time when you feel more hopeful about tomorrow. What's one thing that usually brings you even a tiny bit of joy?",
    
    "Your honesty about your struggles takes real bravery. Like a river that finds its way around obstacles, you have the ability to navigate through this challenging time. Consider challenging negative thoughts by asking 'Is this thought helpful or true?' Imagine feeling more balanced and at peace. What support do you feel you need most right now?",
    
    "I appreciate you opening up about how you're feeling. Your inner strength shines like a candle in the darkness, even when you can't see it yourself. Try the breathing technique: breathe in for 4 counts, hold for 4, breathe out for 6. Picture a day when you feel more connected to hope and possibility. What's one small thing you could do today to care for yourself?"
]

@app.get("/")
def read_root():
    return {"status": "PHQ-9 Chatbot API is running"}

@app.post("/phq9-chat")
def chat(req: ChatRequest):
    # Simulate processing time
    time.sleep(1)
    
    # Get the last user message
    user_messages = [m['content'] for m in req.messages if m['role'] == 'user']
    last_message = user_messages[-1] if user_messages else ""
    
    # Simple keyword-based responses for testing
    if any(word in last_message.lower() for word in ['sad', 'depressed', 'down', 'hopeless']):
        response = "I hear the sadness in your words, and I want you to know that what you're feeling is valid. Your courage to express these feelings is like a small flame of hope that refuses to be extinguished. Try the 'STOP' technique: Stop what you're doing, Take a breath, Observe your thoughts and feelings, Proceed with kindness toward yourself. Imagine a moment when you feel a little lighter. What's one thing that has helped you feel even slightly better in the past?"
    
    elif any(word in last_message.lower() for word in ['anxious', 'worried', 'stress', 'panic']):
        response = "I can feel the anxiety you're experiencing, and it takes strength to reach out when you're feeling this way. Your mind is like a garden that sometimes gets overgrown with worries, but with gentle care, peace can bloom again. Try the 4-7-8 breathing technique: breathe in for 4, hold for 7, breathe out for 8. Picture a time when you feel calm and grounded. What usually helps you feel more centered?"
    
    elif any(word in last_message.lower() for word in ['tired', 'exhausted', 'energy', 'sleep']):
        response = "The exhaustion you're feeling is real, and it's okay to acknowledge how hard things have been. Your spirit is like a battery that needs gentle recharging, not harsh demands. Consider creating a simple evening routine that signals to your body it's time to rest. Envision waking up feeling more refreshed and renewed. What's one small change you could make to your sleep routine?"
    
    elif any(word in last_message.lower() for word in ['alone', 'lonely', 'isolated', 'disconnected']):
        response = "Feeling alone can be one of the most difficult experiences, and I'm grateful you're sharing this with me. Your connection to others is like a bridge that may feel damaged but can be rebuilt, one conversation at a time. Try reaching out to one person today, even with a simple message. Imagine feeling more connected and supported. Who in your life might appreciate hearing from you?"
    
    else:
        # Use a random response for general messages
        response = random.choice(sample_responses)
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)