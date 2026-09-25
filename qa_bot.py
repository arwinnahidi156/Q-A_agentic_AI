"""
Terminal Q&A bot with web search
question -> web_search (Perplexity Sonar on Kaya)
         -> ChatPromptTemplate {question, context, today}
         -> model.with_structured_output(QAResponse) -> QAResponse

- Web search: real, current information with real source URLs
- Structured output: answer + confidence + sources
- Resilience: search-model fallbacks, answer-model fallbacks, parse retry, raw-text rescue
- Graceful failure: if search fails, answer from memory; every error becomes a readable message
- LangSmith tracing: controlled only by env vars (.env)
"""
import os
import sys
import time
import uuid
import logging
from datetime import date
from functools import lru_cache
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import traceable
from langsmith.wrappers import wrap_openai

load_dotenv()


# ------------------------------------------------------------------
# Env helpers
# ------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.split("#")[0].strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.split("#")[0].strip())
    except ValueError:
        return default


def _env_list(name: str, default: str) -> List[str]:
    value = os.getenv(name) or default
    return [item.strip() for item in value.split("#")[0].split(",") if item.strip()]


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
KAYA_BASE_URL = "https://kayaai.ir/api"
MAX_QUESTION_CHARS = 2000
PARSE_ATTEMPTS = 2                          # 1 normal try + 1 retry on bad structure
MODEL_CANDIDATES = [
    "deepseek/deepseek-v4-pro-0813",        # primary
    "openai/gpt-5.6-luna-pro",              # fallback 1
    "meta/muse-spark-1.3-contributor",      # fallback 2
]

DEBUG = _env_bool("QA_DEBUG", True)
VERBOSE = _env_bool("QA_VERBOSE", False)
REQUEST_TIMEOUT = _env_int("QA_TIMEOUT", 60)
MAX_RETRIES = _env_int("QA_MAX_RETRIES", 2)

WEB_SEARCH = _env_bool("QA_WEB_SEARCH", True)
SEARCH_MODELS = _env_list("QA_SEARCH_MODELS", "perplexity/sonar,perplexity/sonar-pro")
SEARCH_TIMEOUT = _env_int("QA_SEARCH_TIMEOUT", 60)   # search is slower than a normal answer
MAX_SOURCES = 5

# Kaya must never go through a proxy/VPN (it rejects foreign IPs)
KAYA_HTTP_CLIENT = httpx.Client(trust_env=False)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("qa_bot")

if DEBUG:
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)

if VERBOSE:
    set_debug(True)


# ------------------------------------------------------------------
# 1. Schema
# ------------------------------------------------------------------
class Source(BaseModel):
    title: str = Field(description="Name of the source, e.g. 'Python docs' or 'Wikipedia: Photosynthesis'")
    url: Optional[str] = Field(default=None, description="URL ONLY if you are certain it exists, otherwise null")


class QAResponse(BaseModel):
    """Structured answer to a user's question."""
    answer: str = Field(description="Clear, concise answer to the question")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence that the answer is correct, from 0.0 to 1.0")
    sources: List[Source] = Field(default_factory=list, description="References supporting the answer. Empty list if none")


class SearchResult(BaseModel):
    """What the web search step returns (never sent to the model as a schema)."""
    text: str
    sources: List[Source] = Field(default_factory=list)
    model: str = ""


# ------------------------------------------------------------------
# 2. Web search (Perplexity Sonar through Kaya)
# ------------------------------------------------------------------
SEARCH_SYSTEM_PROMPT = (
    "Search the web and report the facts that answer the user's question. "
    "Be factual and concise. Include dates, numbers and names where relevant. "
    "Mark claims with citation numbers like 1, 2."
)


@lru_cache(maxsize=1)
def _search_client() -> openai.OpenAI:
    """Raw OpenAI-format client for Kaya. wrap_openai makes each call show up in LangSmith."""
    client = openai.OpenAI(
        api_key=os.environ["KAYA_API_KEY"],
        base_url=KAYA_BASE_URL,
        http_client=KAYA_HTTP_CLIENT,       # direct, no VPN
        timeout=SEARCH_TIMEOUT,
        max_retries=MAX_RETRIES,
    )
    return wrap_openai(client)


def _get(obj, key):
    """Read a field from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _title_for(url: str, title: Optional[str]) -> str:
    return (title or "").strip() or urlparse(url).netloc or url


def _extract_sources(resp) -> List[Source]:
    """
    Search models return citations in different formats depending on the gateway.
    We read all three known formats and de-duplicate by URL, keeping the original order.
    """
    found = {}

    # 1) Perplexity native: search_results = [{"title", "url", "date"}, ...]
    for item in _get(resp, "search_results") or []:
        url = _get(item, "url")
        if url:
            found.setdefault(url, _title_for(url, _get(item, "title")))

    # 2) Perplexity native: citations = ["https://...", ...]
    for url in _get(resp, "citations") or []:
        if isinstance(url, str) and url:
            found.setdefault(url, _title_for(url, None))

    # 3) OpenAI/OpenRouter style: message.annotations = [{"type": "url_citation", "url_citation": {...}}]
    message = resp.choices[0].message if resp.choices else None
    for ann in _get(message, "annotations") or []:
        cit = _get(ann, "url_citation")
        url = _get(cit, "url") if cit else None
        if url:
            found.setdefault(url, _title_for(url, _get(cit, "title")))

    return [Source(title=title, url=url) for url, title in found.items()]


@traceable(name="web_search", run_type="retriever")
def web_search(question: str) -> Optional[SearchResult]:
    """
    Try each search model in order. Returns None if search is unavailable,
    so the bot can still answer from memory. Key/credit errors are re-raised.
    """
    for model in SEARCH_MODELS:
        started = time.perf_counter()
        if DEBUG:
            print(f"   🌐 searching the web with {model} ...", flush=True)
        try:
            resp = _search_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
        except openai.APIStatusError as e:
            if e.status_code in (401, 402):  # wrong key / no credit: other models won't help
                raise
            if DEBUG:
                print(f"   ❌ search failed ({model}): {type(e).__name__} {e.status_code}", flush=True)
            continue
        except openai.APIError as e:         # timeout, connection error, ...
            if DEBUG:
                print(f"   ❌ search failed ({model}): {type(e).__name__}", flush=True)
            continue

        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        sources = _extract_sources(resp)[:MAX_SOURCES]
        if DEBUG:
            secs = time.perf_counter() - started
            print(f"   ✅ search done in {secs:.1f}s | {len(sources)} sources", flush=True)
        if text:
            return SearchResult(text=text, sources=sources, model=model)

    return None


def build_context(search: Optional[SearchResult]) -> str:
    if search is None:
        return "WEB RESULTS: none. Answer from your own knowledge."
    numbered = "\n".join(f"[{i}] {s.title} - {s.url}" for i, s in enumerate(search.sources, 1))
    return (
        "WEB RESULTS:\n"
        f"{search.text}\n\n"
        "SOURCES:\n"
        f"{numbered or '(the search returned no URLs)'}"
    )


# ------------------------------------------------------------------
# 3. Prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise question-answering assistant. Today's date is {today}.
Rules:
- Answer directly and concisely.
- If WEB RESULTS are provided, base your answer on them first. They are more current than
  your own knowledge. Keep citation markers like 1, 2 that match the SOURCES list.
  If the web results don't answer the question, say so.
- If no WEB RESULTS are provided, answer from your own knowledge and LOWER the confidence
  for anything that may have changed recently.
- confidence scale: 0.9-1.0 well-established fact | 0.6-0.8 likely, some uncertainty |
  0.3-0.5 partial/uncertain | below 0.3 mostly guessing.
- If the question is ambiguous or unanswerable, say so in the answer and LOWER the confidence.
- sources: when WEB RESULTS are provided, return an empty list (the real URLs are attached
  automatically). Otherwise cite only real, well-known references and NEVER invent URLs.
- Reply in the same language as the question."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\n{context}"),
])


# ------------------------------------------------------------------
# 4. Model + chain
# ------------------------------------------------------------------
def get_model(name: str,
              temperature: float = 0.2,
              max_tokens: int = 1500,
              timeout: int = REQUEST_TIMEOUT,
              max_retries: int = MAX_RETRIES):
    return init_chat_model(
        model=name,
        model_provider="openai",            # Kaya speaks the OpenAI API format
        api_key=os.environ["KAYA_API_KEY"],
        base_url=KAYA_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        http_client=KAYA_HTTP_CLIENT,       # direct, no VPN
    )


def build_chain():
    structured_models = [
        get_model(name).with_structured_output(
            QAResponse,
            method="function_calling",
            include_raw=True,
        )
        for name in MODEL_CANDIDATES
    ]
    llm = structured_models[0].with_fallbacks(structured_models[1:])
    return prompt | llm


# ------------------------------------------------------------------
# 5. Live step printer (debug)
# ------------------------------------------------------------------
class StepPrinter(BaseCallbackHandler):
    """Prints what the LLM is doing, live, in the terminal."""

    def __init__(self):
        self._start = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        meta = kwargs.get("metadata") or {}
        params = kwargs.get("invocation_params") or {}
        name = params.get("model") or params.get("model_name") or meta.get("ls_model_name", "model")
        self._start[run_id] = time.perf_counter()
        print(f"   ⏳ answering with {name} ...", flush=True)

    def on_llm_end(self, response, *, run_id, **kwargs):
        secs = time.perf_counter() - self._start.pop(run_id, time.perf_counter())
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None) or {}
            tool = "tool call ✅" if getattr(msg, "tool_calls", None) else "NO tool call ⚠️"
            tokens = usage.get("total_tokens", "?")
        except Exception:
            tool, tokens = "?", "?"
        print(f"   ✅ reply in {secs:.1f}s | {tool} | tokens: {tokens}", flush=True)

    def on_llm_error(self, error, *, run_id, **kwargs):
        secs = time.perf_counter() - self._start.pop(run_id, time.perf_counter())
        print(f"   ❌ failed after {secs:.1f}s: {type(error).__name__}: {error}", flush=True)


STEP_PRINTER = StepPrinter()


# ------------------------------------------------------------------
# 6. Core logic (one traced "turn" in LangSmith)
# ------------------------------------------------------------------
def _message_text(message) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content).strip()


@traceable(
    name="qa_turn",
    run_type="chain",
    tags=["qa-bot", "terminal"],
    process_inputs=lambda inputs: {"question": inputs.get("question")},
)
def ask(chain, question: str) -> QAResponse:
    # Step 1: search (optional, never fatal)
    search = web_search(question) if WEB_SEARCH else None
    if WEB_SEARCH and search is None and DEBUG:
        print("   ⚠️  web search unavailable, answering from memory", flush=True)

    # Step 2: structured answer grounded in the search results
    inputs = {
        "question": question,
        "context": build_context(search),
        "today": date.today().isoformat(),
    }
    config = {"run_name": "qa_chain"}
    if DEBUG:
        config["callbacks"] = [STEP_PRINTER]

    resp = None
    result = None
    for attempt in range(1, PARSE_ATTEMPTS + 1):
        result = chain.invoke(inputs, config=config)
        if result.get("parsed") is not None:
            resp = result["parsed"]
            break
        log.warning("Structured parse failed (attempt %d): %s", attempt, result.get("parsing_error"))

    # Salvage: model replied, but not in the schema -> raw text, or the search text itself
    if resp is None:
        raw_text = _message_text(result.get("raw")) if result else ""
        fallback = raw_text or (search.text if search else "")
        resp = QAResponse(
            answer=fallback or "I couldn't produce a structured answer for that question.",
            confidence=0.0,
        )

    # Step 3: real URLs from the search replace anything the model remembered
    if search and search.sources:
        resp.sources = search.sources

    return resp


def safe_ask(chain, question: str, session_id: str) -> Tuple[Optional[QAResponse], Optional[str]]:
    """Graceful failure layer: turns every exception into a readable message."""
    extra = {"metadata": {"session_id": session_id}}
    try:
        return ask(chain, question, langsmith_extra=extra), None
    except openai.AuthenticationError:
        return None, "Authentication failed. Check KAYA_API_KEY."
    except openai.RateLimitError:
        return None, "Rate limit hit (Kaya allows 60 requests/min, 10 when credit is low). Wait a minute."
    except (openai.APITimeoutError, openai.APIConnectionError):
        return None, "Network/timeout problem reaching Kaya. Check your connection."
    except openai.APIStatusError as e:
        if e.status_code == 402:
            return None, "Kaya credit is too low. Top up your account."
        return None, f"Provider returned an error ({e.status_code}). Try again later."
    except Exception as e:
        log.exception("Unexpected error")
        return None, f"Unexpected error: {type(e).__name__}"


# ------------------------------------------------------------------
# 7. Terminal UI
# ------------------------------------------------------------------
def validate(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = question.strip()
    if not q:
        return None, "Please type a question."
    if len(q) > MAX_QUESTION_CHARS:
        return None, f"Question too long ({len(q)} chars, max {MAX_QUESTION_CHARS})."
    return q, None


def render(resp: QAResponse) -> None:
    filled = round(resp.confidence * 10)
    bar = "█" * filled + "░" * (10 - filled)
    print(f"\n📝 Answer:\n{resp.answer}\n")
    print(f"🎯 Confidence: {bar} {resp.confidence:.0%}")
    if resp.sources:
        print("📚 Sources:")
        for i, s in enumerate(resp.sources, 1):
            print(f"   [{i}] {s.title}" + (f" | {s.url}" if s.url else ""))
    else:
        print("📚 Sources: none cited")
    print("-" * 60, flush=True)


def main():
    if not os.getenv("KAYA_API_KEY"):
        sys.exit("❌ KAYA_API_KEY is missing from .env")

    tracing_on = _env_bool("LANGSMITH_TRACING", False)
    project = os.getenv("LANGSMITH_PROJECT", "default")
    print(f"🔍 LangSmith tracing: {'ON → project ' + project if tracing_on else 'OFF'}")
    print(f"🌐 Web search: {'ON → ' + ', '.join(SEARCH_MODELS) if WEB_SEARCH else 'OFF'}")
    if DEBUG:
        print(f"🛠  Debug: ON | timeout={REQUEST_TIMEOUT}s | search timeout={SEARCH_TIMEOUT}s | retries={MAX_RETRIES}")
        print(f"🧠 Answer models (in order): {', '.join(MODEL_CANDIDATES)}")

    chain = build_chain()
    session_id = str(uuid.uuid4())
    print("🤖 Ask me anything. Type 'exit' to quit.\n", flush=True)

    try:
        while True:
            try:
                raw = input("❓ You: ")
            except EOFError:
                print()
                break
            if raw.strip().lower() in {"exit", "quit", "q"}:
                break

            question, err = validate(raw)
            if err:
                print(f"⚠️  {err}\n", flush=True)
                continue

            print("🤔 Thinking...", flush=True)
            started = time.perf_counter()
            resp, err = safe_ask(chain, question, session_id)
            if DEBUG:
                print(f"   ⏱  total: {time.perf_counter() - started:.1f}s", flush=True)

            if err:
                print(f"⚠️  {err}\n", flush=True)
                continue
            render(resp)
    except KeyboardInterrupt:
        print()
    finally:
        if tracing_on:
            print("📤 Sending remaining traces to LangSmith...", flush=True)
        wait_for_all_tracers()
        print("👋 Bye!")


if __name__ == "__main__":
    main()