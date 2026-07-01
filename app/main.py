"""
SHL Assessment Recommender — FastAPI Service
Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse
"""
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.models import ChatRequest, ChatResponse
from app import retrieval, agent

load_dotenv()

_startup_pool = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the retrieval index in the background so /health responds immediately."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[startup] WARNING: GROQ_API_KEY is not set — /chat will not work.")
    else:
        print("[startup] GROQ_API_KEY detected.")

    print("[startup] Warming TF-IDF index in background...")
    _startup_pool.submit(retrieval.initialize)
    print("[startup] Ready.")
    yield
    _startup_pool.shutdown(wait=False)


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    """Root endpoint for basic service status."""
    return {
        "service": "SHL Assessment Recommender",
        "status": "ok",
        "health": "/health",
        "chat": "/chat",
    }


@app.get("/health")
def health() -> dict:
    """Health check endpoint — must respond quickly for Render."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.
    The full conversation history is passed in each request.
    """
    if not request.messages:
        return ChatResponse(
            reply="Hello! I can help you find the right SHL assessments. "
                  "What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False
        )

    # Validate roles
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=422, detail=f"Invalid role: {msg.role}")

    try:
        reply, recommendations, end_of_conversation = agent.process(request.messages)
    except Exception as e:
        print(f"[agent error] {type(e).__name__}: {e}")
        return ChatResponse(
            reply="I encountered an issue processing your request. Could you rephrase your query?",
            recommendations=[],
            end_of_conversation=False
        )

    # Hard guardrail: cap recommendations at 10
    recommendations = recommendations[:10]

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )
