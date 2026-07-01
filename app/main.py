"""
SHL Assessment Recommender — FastAPI Service
Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse
"""
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.models import ChatRequest, ChatResponse
from app import retrieval, agent

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize retrieval index on startup."""
    print("[startup] Building FAISS index...")
    retrieval.initialize()
    print("[startup] Ready.")
    yield


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


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
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
    except (json.JSONDecodeError, RuntimeError, ValueError) as e:
        print(f"[agent error] {e}")
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
