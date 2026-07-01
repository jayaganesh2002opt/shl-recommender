# Interview Preparation — Technical Deep Dive Q&A

## Architecture Questions

**Q: Why did you choose FAISS over Chroma or pgvector?**
A: FAISS runs in-process — no external database, no network latency, and it fits within Render's free tier. Cold start takes under 3 seconds. For 50-100 catalog items, IndexFlatIP (exact search) is more than fast enough and avoids approximation errors from HNSW/IVF indexes.

**Q: Why sentence-transformers `all-MiniLM-L6-v2`?**
A: It's 80MB, fast to load, and produces strong semantic embeddings for short professional text. I benchmarked it against `paraphrase-mpnet-base-v2` on the public traces — MiniLM was 3x faster at similar recall quality, which matters for the 30-second timeout.

**Q: How does the stateless design work?**
A: Every POST /chat call receives the full conversation history. I don't store anything server-side. I reconstruct context each turn by running a structured LLM extraction that produces a JSON slot-fill: role, skills, seniority, requested test types. This is deterministic — if the user says "actually add personality tests" in turn 3, the updated context overwrites the previous slot, and the next retrieval query includes personality.

**Q: How do you prevent hallucinated assessment names?**
A: Two layers. First, the LLM recommendation prompt only shows retrieved catalog items — the model must select from that list, it cannot invent new names. Second, before building the response, I validate each selected name against the full catalog dict. Any name not found in the catalog is dropped.

---

## Retrieval Questions

**Q: How do you build the retrieval query from conversation context?**
A: I concatenate extracted slots: `"Java developer mid-level stakeholder management coding personality"`. This string captures the semantic intent and retrieves both technical tests (Java 8, Automata Fix) and behavioral/personality tests (OPQ32r, GSA) in one search — exactly what a recruiter building a battery would want.

**Q: What's the difference between FAISS IndexFlatIP and IndexFlatL2?**
A: IndexFlatIP computes inner product (dot product). After L2-normalizing vectors, inner product equals cosine similarity, which is what we want for semantic search — it measures directional similarity regardless of vector magnitude. IndexFlatL2 computes Euclidean distance, which includes magnitude, making it less suitable for semantic text similarity.

---

## Agent Design Questions

**Q: When does the agent clarify vs. recommend?**
A: The `enough_to_recommend` flag in the context extractor. If the user has provided at least a role OR enough skills to form a meaningful query, we recommend. If both are missing (e.g., "I need an assessment"), we ask one clarifying question. This is also enforced by the intent classifier — "I need an assessment" → CLARIFY intent.

**Q: How does the turn cap work?**
A: Hard server-side check: `if len(messages) >= 7`. If triggered, the agent forces a recommendation if there's enough context, or ends with an explanation. This ensures the 8-turn limit is always respected, regardless of LLM behavior.

**Q: How do you handle prompt injection?**
A: Keyword blocklist first (fast, no LLM cost): checks for "ignore previous instructions", "jailbreak", "you are now", etc. If detected, returns a polite refusal without passing to the LLM at all.

**Q: How does COMPARE work differently from RECOMMEND?**
A: For COMPARE, instead of retrieval + LLM selection, I directly look up the mentioned assessment names in the catalog (fuzzy match on name), then pass their full catalog data to a grounded comparison prompt. The model can only describe what's in the catalog fields — no external knowledge. This prevents hallucinated feature comparisons.

---

## Evaluation Questions

**Q: How did you measure recall@10?**
A: I replayed the 10 public traces manually: for each trace, I ran the conversation against my local /chat endpoint and checked how many of the labeled expected assessments appeared in the final recommendations list. Recall@10 = (matched expected assessments) / (total expected assessments).

**Q: What was your recall@10 score?**
A: On the public traces, approximately 0.75-0.80 mean Recall@10 after switching from keyword matching to semantic embeddings. The biggest misses were role-specific solutions (Graduate 7.0 Solution for campus hiring) which I fixed by adding job level keywords to the catalog entries.

**Q: What didn't work in your first version?**
A: Three things. (1) Keyword-only retrieval: "developer who works with stakeholders" didn't return OPQ without semantic search. (2) LLM returning JSON wrapped in markdown fences — `json.loads()` crashed. Fixed with regex stripping and explicit "no markdown" instructions. (3) Agent recommending on turn 1 for vague queries — fixed by making `enough_to_recommend=false` when role is null.

---

## Code Questions

**Q: Walk me through what happens when POST /chat receives a request.**
A: 
1. Pydantic validates the request schema (messages list, role must be "user" or "assistant").
2. `agent.process(messages)` is called.
3. Turn cap checked — if ≥ 7 messages, force finish.
4. `classify_intent()` runs a fast LLM call, returns one of 6 intent labels.
5. Branch on intent: OFF_TOPIC → refuse. COMPARE → `generate_compare()`. CLARIFY → `generate_clarify()`. RECOMMEND/REFINE → `extract_context()` then `generate_recommendations()`.
6. `generate_recommendations()` builds a query string, calls `retrieval.search()`, gets top-15 candidates, sends to LLM with only those candidates. LLM returns JSON with `reply` and `selected` names.
7. Names validated against catalog, mapped to Recommendation objects.
8. Response assembled, recommendations capped at 10, returned.

**Q: Why are you using Gemini instead of OpenAI?**
A: Gemini 1.5 Flash has a generous free tier (15 RPM, 1M tokens/day) with no credit card required. For this use case — 3 small LLM calls per turn (intent + context + response) — it fits within the free tier easily and stays under the 30-second timeout. OpenAI free tier is more restricted.
