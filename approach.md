# Approach Document — SHL Assessment Recommender

## Problem Decomposition

The core challenge is bridging the gap between a recruiter's vague, natural-language hiring intent and a structured catalog of SHL assessments. I decomposed this into four sub-problems: (1) catalog ingestion and retrieval, (2) conversation state management without server-side persistence, (3) intent-aware agent behavior, and (4) scope enforcement.

---

## Design Choices

### Catalog Ingestion
I scraped the SHL Individual Test Solutions catalog (`type=1`) using `requests` + `BeautifulSoup` with a paginated loop (12 items/page). Since the live catalog page is JavaScript-rendered, I also wrote a Playwright fallback. The scraped data is stored as `catalog.json` — a flat list of objects with `name`, `url`, `test_type`, `description`, `job_levels`, `keywords`, `duration_minutes`, and `remote_testing` flags. This structured representation lets the agent both retrieve and compare assessments entirely from grounded data.

### Retrieval: Semantic Search over FAISS
I embedded each catalog item's combined text (name + description + job levels + keywords) using `sentence-transformers` (`all-MiniLM-L6-v2`) and indexed them in a FAISS `IndexFlatIP` with L2-normalized vectors for cosine similarity. On every recommendation turn, I build a query string from extracted context (role + skills + seniority + requested test types) and retrieve the top-15 candidates. This outperforms keyword search because terms like "developer who works with stakeholders" correctly surfaces both technical (Java) and behavioral (OPQ) tests.

**Why FAISS over Chroma/pgvector:** Zero infrastructure — FAISS runs in-process, no external DB, fits within Render's free tier memory, and cold starts in < 3 seconds.

### Stateless State Management
Every `/chat` call receives the full conversation history. I reconstruct context by running a fast structured-extraction LLM call each turn, producing a JSON with fields: `role`, `skills`, `seniority`, `test_types_wanted`, `industry`, `enough_to_recommend`. This slot-filling approach means refinements ("add personality tests") naturally override prior slots without restarting — the updated slots flow into the next retrieval query.

### Agent Intent Classification
A single LLM classifier maps the last user message + recent history to one of: `CLARIFY`, `RECOMMEND`, `REFINE`, `COMPARE`, `OFF_TOPIC`, `GREETING`. This drives deterministic branching:
- **CLARIFY**: ask one focused question (role? seniority?)
- **RECOMMEND/REFINE**: extract context → semantic search → grounded LLM selection
- **COMPARE**: retrieve named items from catalog, generate comparison from catalog data only
- **OFF_TOPIC**: refuse politely

### Scope Enforcement
Two layers: (1) a keyword blocklist catches obvious off-topic/injection attempts before the LLM call, (2) the LLM is constrained to select only names from the retrieved candidate list — it cannot invent assessment names. Every URL in the response is validated against the scraped catalog before being returned.

### Turn Cap
A hard server-side counter: if `len(messages) >= 7`, the agent forces a recommendation or ends the conversation. This ensures the 8-turn limit is always honored regardless of LLM behavior.

---

## Prompt Design

Three specialized prompts keep token usage low and latency under 30 seconds:
1. **Context extractor** (~200 tokens) — structured JSON extraction, strict JSON-only output
2. **Intent classifier** (~150 tokens) — single-label output
3. **Recommendation generator** (~600 tokens) — grounded on retrieved items, JSON output with `reply` + `selected` fields

All prompts use system-level instruction to return JSON only (no markdown fences), then strip any stray formatting before `json.loads()`.

---

## Evaluation Approach

I tested against the 10 public traces by replaying them manually and measuring:
- **Recall@10**: Did the labeled expected assessments appear in the top-10 recommendations?
- **Behavior probes**: Does the agent clarify on turn 1 for vague queries? Does it refuse off-topic? Does it honor mid-conversation refinements?
- **Hallucination check**: Are all returned URLs present in `catalog.json`?

**What didn't work initially:** Pure keyword matching for retrieval (e.g., "stakeholder management" didn't match OPQ without semantic embeddings). Switching to sentence-transformers improved recall from ~0.55 to ~0.78 on the public traces. Also, early prompts returned markdown-wrapped JSON which crashed `json.loads()` — fixed by stripping fences and adding explicit "no markdown" instructions.

---

## Stack Summary

| Component | Choice | Reason |
|-----------|--------|--------|
| API framework | FastAPI + Pydantic | Fast, schema validation, OpenAPI docs |
| LLM | Gemini 1.5 Flash | Free tier, fast (< 5s/call), good instruction following |
| Embeddings | all-MiniLM-L6-v2 | Fast, small (80MB), strong semantic quality |
| Vector store | FAISS IndexFlatIP | In-process, no infra, cold start < 3s |
| Deployment | Render free tier | Zero-cost, Docker/Git deploy, persistent disk |

**AI tools used:** Claude (Anthropic) for code scaffolding and prompt drafting. All design decisions and prompt logic reviewed and understood manually. Code tested locally against the public traces before submission.
