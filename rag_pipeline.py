import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from google import generativeai as genai
from dotenv import load_dotenv

# ==============================
# Load environment and models
# ==============================
load_dotenv()
FAISS_PATH = "index.faiss"
PKL_PATH = "index.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "models/gemini-2.5-flash"

print("Loading models and data...")

# Embedding + FAISS
embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(FAISS_PATH)
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)
chunks = data["chunks"]
sources = data["sources"]

# Gemini setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(GEMINI_MODEL)

# Emotion detection pipeline (lightweight)
print("Loading emotion detector...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def detect_emotion(text):
    """Return top detected emotion for a given text."""
    try:
        result = emotion_classifier(text)[0][0]
        return result["label"].lower()
    except Exception:
        return "neutral"

# ==============================
# Retrieval Function
# ==============================
def retrieve(query, top_k=4):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    results = [chunks[i] for i in I[0]]
    return results

# ==============================
# Chat Logic
# ==============================
chat_history = []

def generate_answer(query):
    q = query.lower()

    # Fixed custom responses
    if "your name" in q or "who are you" in q:
        return "I’m Marcus — your mental health companion and listener 😊"
    elif "who created you" in q or "who made you" in q:
        return "I was created by Lakshya Singh as part of a project to support emotional well-being 💙"
    elif "what can you do" in q:
        return "I can listen, reflect, and help you explore your thoughts with kindness. I'm not a therapist, but I can help you feel heard. 🌿"
    elif "help me" in q or "i need support" in q:
        return "I'm here for you. Please share what's on your mind, and I'll do my best to support you. 💚 Remember, reaching out to a mental health professional can also be very helpful."
    elif "thank you" in q or "thanks" in q:
        return "You're very welcome! I'm here whenever you need to talk. Take care of yourself! 🌼"

    # RAG context retrieval
    context = "\n\n".join(retrieve(query))

    # Recent chat history
    history_text = ""
    if chat_history:
        history_text = "\n".join([
            f"User: {h['user']}\nMarcus: {h['bot']}"
            for h in chat_history[-3:]
        ])

    # 🧠 Emotion detection
    emotion = detect_emotion(query)
    print(f"[Emotion detected: {emotion}]")

    # Build final prompt
    prompt = f"""
You are Marcus, a kind, empathetic, and non-judgmental mental health support assistant.
You remember the flow of recent chats, so you respond naturally and with compassion.

Detected user emotion: {emotion}

Recent chat history:
{history_text}

Knowledge context (from RAG DB):
{context}

User: {query}

Respond with warmth and understanding that fits the user's emotional tone.
Do NOT give medical advice or diagnosis.
If the user sounds in crisis, gently suggest reaching out to a professional or helpline.
    """

    # Generate Gemini response
    response = model.generate_content(prompt)
    answer = response.text.strip()

    # Save chat history
    chat_history.append({"user": query, "bot": answer})

    return answer

# ==============================
# Run chatbot in console
# ==============================
if __name__ == "__main__":
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        print("\nBot:", generate_answer(query))
