"""
Agent module: processes conversation history, classifies intent,
retrieves catalog items, and generates grounded responses.
"""
import os
import json
import re
from typing import List, Dict, Any, Tuple

from groq import Groq

from app import retrieval
from app.models import Message, Recommendation

# Configure Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")

if GROQ_API_KEY:
    _client = Groq(api_key=GROQ_API_KEY)
else:
    _client = None


def _ensure_client():
    if _client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set it in Render environment variables."
        )
    return _client

# ── Scope guard ───────────────────────────────────────────────────────────────
OFF_TOPIC_KEYWORDS = [
    "lawsuit", "legal", "discrimination", "gdpr", "compliance",
    "ignore previous", "ignore all", "disregard", "jailbreak",
    "system prompt", "you are now", "act as", "forget your instructions",
    "interview tips", "resume", "cv", "salary negotiation",
    "stock price", "weather", "recipe", "covid", "politics",
]

def _is_off_topic(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in OFF_TOPIC_KEYWORDS)


# ── Context extraction ────────────────────────────────────────────────────────
_CONTEXT_PROMPT = """You are a context extractor for an SHL assessment recommender.
Given the conversation below, extract hiring context as JSON with these fields:
- role: job title or role (string or null)
- skills: list of technical skills / domains mentioned (list of strings)
- seniority: job level like "entry", "graduate", "mid", "senior", "manager", "director", "executive" (string or null)
- test_types_wanted: explicitly requested test types, e.g. ["personality", "cognitive", "coding"] (list of strings, may be empty)
- industry: industry or domain if mentioned (string or null)
- additional_context: any other relevant hiring context (string or null)
- enough_to_recommend: true if you have at least a role or enough skills to recommend assessments, false otherwise

Respond ONLY with valid JSON. No markdown, no explanation.

Conversation:
{conversation}
"""

def extract_context(messages: List[Message]) -> Dict[str, Any]:
    conversation = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
    prompt = _CONTEXT_PROMPT.format(conversation=conversation)
    try:
        resp = _ensure_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        # Strip possible markdown fences
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"[extract_context error] {type(e).__name__}: {e}")
        return {"enough_to_recommend": False}


# ── Intent classification ─────────────────────────────────────────────────────
_INTENT_PROMPT = """Classify the LAST user message in this conversation into ONE of:
- CLARIFY: user wants an assessment but hasn't given enough info yet OR the system needs more info
- RECOMMEND: user has given enough context and wants recommendations
- REFINE: user is modifying/adding constraints to a previous recommendation
- COMPARE: user wants to compare two or more named assessments
- OFF_TOPIC: user is asking something unrelated to SHL assessments
- GREETING: greeting or very generic opener

Respond with ONLY the intent label. Nothing else.

Last user message: {last_message}
Full conversation so far:
{conversation}
"""

def classify_intent(messages: List[Message]) -> str:
    if not messages:
        return "CLARIFY"
    last = messages[-1].content
    if _is_off_topic(last):
        return "OFF_TOPIC"
    conversation = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages[-6:])
    prompt = _INTENT_PROMPT.format(last_message=last, conversation=conversation)
    try:
        resp = _ensure_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        intent = resp.choices[0].message.content.strip().upper()
        for valid in ["CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "OFF_TOPIC", "GREETING"]:
            if valid in intent:
                return valid
        return "CLARIFY"
    except Exception as e:
        print(f"[classify_intent error] {type(e).__name__}: {e}")
        return "CLARIFY"


# ── Clarifying question generator ─────────────────────────────────────────────
_CLARIFY_PROMPT = """You are a helpful SHL assessment recommender assistant.
The user wants to find SHL assessments but hasn't given enough information.
Based on the conversation, ask ONE focused clarifying question to gather the most important missing information.
Keep it short (1-2 sentences). Do not recommend any assessments yet.
Focus on: what role/job are they hiring for, what level (graduate/mid/senior/manager), or what skills matter.

Conversation:
{conversation}

Ask ONE clarifying question:"""

def generate_clarify(messages: List[Message]) -> str:
    conversation = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages[-6:])
    prompt = _CLARIFY_PROMPT.format(conversation=conversation)
    try:
        resp = _ensure_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Could you tell me more about the role you're hiring for and the seniority level?"


# ── Recommendation generator ──────────────────────────────────────────────────
_RECOMMEND_PROMPT = """You are an SHL assessment recommender. Based on the context and retrieved assessments below,
select the most relevant assessments (between 1 and 10) and write a brief, helpful reply.

Hiring context:
{context}

Retrieved SHL catalog assessments (use ONLY these, do not invent others):
{catalog_items}

Instructions:
- Select assessments that best match the role, skills, seniority, and any requested test types.
- Write a 2-3 sentence reply explaining why these assessments fit.
- Return a JSON object ONLY with keys "reply" (string) and "selected" (list of assessment names from the catalog).
- "selected" must be names EXACTLY as they appear in the catalog above.
- Do NOT invent or hallucinate any assessment names.
- Respond ONLY with valid JSON. No markdown fences.
"""

def generate_recommendations(
    messages: List[Message],
    context: Dict[str, Any]
) -> Tuple[str, List[Recommendation]]:
    # Build query from context
    query_parts = []
    if context.get("role"):
        query_parts.append(context["role"])
    if context.get("skills"):
        query_parts.extend(context["skills"])
    if context.get("seniority"):
        query_parts.append(context["seniority"])
    if context.get("test_types_wanted"):
        query_parts.extend(context["test_types_wanted"])
    if context.get("industry"):
        query_parts.append(context["industry"])

    query = " ".join(query_parts) if query_parts else "general professional assessment"
    candidates = retrieval.search(query, top_k=15)

    catalog_text = "\n".join(
        f"- Name: {c['name']}\n  URL: {c['url']}\n  Type: {','.join(c.get('test_type', []))}\n  Description: {c['description'][:200]}"
        for c in candidates
    )

    prompt = _RECOMMEND_PROMPT.format(
        context=json.dumps(context, ensure_ascii=False),
        catalog_items=catalog_text
    )

    try:
        resp = _ensure_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip()
        # Strip possible markdown fences
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```$", "", raw)
        data = json.loads(raw)
        reply = data.get("reply", "Here are my recommendations.")
        selected_names = data.get("selected", [])
    except Exception:
        reply = "Here are some assessments that match your requirements."
        selected_names = [c["name"] for c in candidates[:5]]

    # Map selected names back to catalog items
    recs = []
    name_to_item = {c["name"]: c for c in candidates}
    all_catalog = retrieval.get_all()
    full_name_map = {c["name"]: c for c in all_catalog}

    seen = set()
    for name in selected_names[:10]:
        item = name_to_item.get(name) or full_name_map.get(name)
        if item and item["name"] not in seen:
            seen.add(item["name"])
            recs.append(Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=",".join(item.get("test_type", ["K"]))
            ))

    return reply, recs


# ── Compare generator ─────────────────────────────────────────────────────────
_COMPARE_PROMPT = """You are an SHL assessment expert. Compare the following assessments using ONLY the catalog data provided.
Do not use any outside knowledge. Be factual and concise (3-5 sentences).

Assessments to compare (from SHL catalog):
{items}

User question: {question}

Write a grounded comparison:"""

def generate_compare(messages: List[Message]) -> str:
    last_msg = messages[-1].content if messages else ""
    all_catalog = retrieval.get_all()
    catalog_map = {c["name"].lower(): c for c in all_catalog}

    # Find assessment names mentioned
    mentioned = []
    for item in all_catalog:
        if item["name"].lower() in last_msg.lower():
            mentioned.append(item)

    if len(mentioned) < 2:
        # Fuzzy: try keyword match
        words = last_msg.lower().split()
        for item in all_catalog:
            name_words = item["name"].lower().split()
            if any(w in last_msg.lower() for w in name_words if len(w) > 3):
                if item not in mentioned:
                    mentioned.append(item)

    if not mentioned:
        return "I couldn't identify which assessments you'd like to compare from the catalog. Could you specify the exact assessment names?"

    items_text = "\n\n".join(
        f"**{item['name']}**\n- Type: {','.join(item.get('test_type', []))}\n"
        f"- Description: {item['description']}\n"
        f"- Job levels: {', '.join(item.get('job_levels', []))}\n"
        f"- Duration: {item.get('duration_minutes', 'N/A')} minutes"
        for item in mentioned[:4]
    )

    prompt = _COMPARE_PROMPT.format(items=items_text, question=last_msg)
    try:
        resp = _ensure_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Here is what I know from the catalog:\n\n{items_text}"


# ── Main entry point ──────────────────────────────────────────────────────────
def process(messages: List[Message]) -> Tuple[str, List[Recommendation], bool]:
    """
    Returns (reply, recommendations, end_of_conversation)
    """
    if not messages:
        return (
            "Hello! I can help you find the right SHL assessments. What role are you hiring for?",
            [],
            False
        )

    # Hard turn cap — if 7+ messages, force recommendation or end
    if len(messages) >= 7:
        context = extract_context(messages)
        if context.get("enough_to_recommend"):
            reply, recs = generate_recommendations(messages, context)
            return reply, recs, True
        else:
            return (
                "I need to wrap up. Based on our conversation, I wasn't able to gather enough context. "
                "Please restart with a specific role and seniority level.",
                [],
                True
            )

    # Classify intent
    intent = classify_intent(messages)

    if intent == "OFF_TOPIC":
        return (
            "I'm only able to help with SHL assessment recommendations. "
            "I can't assist with legal questions, general hiring advice, or unrelated topics. "
            "What role are you looking to hire for?",
            [],
            False
        )

    if intent == "GREETING":
        return (
            "Hello! I'm here to help you find the right SHL assessments. "
            "Could you tell me what role you're hiring for?",
            [],
            False
        )

    if intent == "COMPARE":
        reply = generate_compare(messages)
        return reply, [], False

    if intent == "CLARIFY":
        reply = generate_clarify(messages)
        return reply, [], False

    if intent in ("RECOMMEND", "REFINE"):
        context = extract_context(messages)
        if not context.get("enough_to_recommend", False):
            reply = generate_clarify(messages)
            return reply, [], False
        reply, recs = generate_recommendations(messages, context)
        # End conversation when we return a shortlist
        end = len(recs) > 0
        return reply, recs, end

    # Fallback
    reply = generate_clarify(messages)
    return reply, [], False
