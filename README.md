# SHL Assessment Recommender

Conversational AI agent that recommends SHL assessments based on hiring context.

## Quick Setup

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd shl-recommender
pip install -r requirements.txt
```

### 2. Set Environment Variable
Get a free Gemini API key from https://aistudio.google.com/app/apikey

```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. (Optional) Re-scrape Catalog
```bash
python scripts/scrape_catalog.py
```
A pre-built `data/catalog.json` is already included.

### 4. Run Locally
```bash
uvicorn app.main:app --reload --port 8000
```

### 5. Test
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"}
    ]
  }'
```

## Deploy to Hugging Face Spaces (Free)

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Select "Docker" as Space SDK
4. Name it: `shl-recommender`
5. Make it Public
6. Clone the Space locally:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/shl-recommender
   cd shl-recommender
   ```
7. Copy all files from this repo to the Space
8. Commit and push:
   ```bash
   git add .
   git commit -m "Initial deploy"
   git push
   ```
9. Go to your Space settings → Repository → Secrets
10. Add secret: `GEMINI_API_KEY` = your_key
11. The Space will automatically rebuild

## Deploy to Render (Free)

1. Push this repo to GitHub
2. Go to https://render.com → New Web Service → Connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GEMINI_API_KEY = your_key`
6. Deploy!

## API Reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
Request:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response:
```json
{
  "reply": "Here are 5 assessments...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Architecture
- **FastAPI** — REST API, Pydantic schema validation
- **Gemini Flash** — LLM for intent classification, context extraction, response generation
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embedding catalog items
- **FAISS** — vector similarity search over catalog
- **Stateless design** — full conversation history passed per request
