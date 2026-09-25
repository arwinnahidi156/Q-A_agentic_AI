"""
Terminal Q&A bot with web search and conversation memory
question -> (rewrite follow-up into a standalone search query)
         -> web_search (Perplexity Sonar on Kaya)
         -> ChatPromptTemplate {system, history, question, context, today}
         -> model.with_structured_output(QAResponse) -> QAResponse
         -> save (question, answer) to short-term memory

Commands: /history  show memory | /reset  clear memory | exit  quit
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
from pydantic import BaseModel, Field, PrivateAttr
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_debug
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
PARSE_ATTEMPTS = 2
MODEL_CANDIDATES = [
    "deepseek/deepseek-v4-pro-0813",        # primary
    "openai/gpt-5.6-luna-pro",              # fallback 1
    "meta/muse-spark-1.3-contributor",      # fallback 2
]

DEBUG = _env_bool("QA_DEBUG", True)
VERBOSE = _env_bool("QA_VERBOSE", False)
REQUEST_TIMEOUT = _env_int("QA_TIMEOUT", 60)
MAX_RETRIES = _env_int("QA_MAX_RETRIES", 1)

WEB_SEARCH = _env_bool("QA_WEB_SEARCH", True)
SEARCH_MODELS = _env_list("QA_SEARCH_MODELS", "perplexity/sonar,perplexity/sonar-pro")
SEARCH_TIMEOUT = _env_int("QA_SEARCH_TIMEOUT", 60)
MAX_SOURCES = 5

MEMORY_TURNS = _env_int("QA_MEMORY_TURNS", 6)   # question/answer pairs kept; 0 = no memory

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
    # Private: not part of the schema, the model never sees or fills these
    _answered_by: str = PrivateAttr(default="")
    _searched_with: str = PrivateAttr(default="")


class SearchResult(BaseModel):
    """What the web search step returns (never sent to the model as a schema)."""
    text: str
    sources: List[Source] = Field(default_factory=list)
    model: str = ""


# ------------------------------------------------------------------
# 2. Short-term memory
# ------------------------------------------------------------------
class ConversationMemory:
    """Keeps the last N question/answer turns in RAM (lost when the program exits)."""

    def __init__(self, max_turns: int):
        self.max_turns = max(0, max_turns)
        self._messages: List[BaseMessage] = []

    def add(self, question: str, answer: str) -> None:
        self._messages += [HumanMessage(content=question), AIMessage(content=answer)]
        excess = len(self._messages) - self.max_turns * 2
        if excess > 0:
            del self._messages[:excess]         # drop the oldest turns

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._messages)             # a copy, so callers can't change the memory

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages) // 2         # number of turns


# ------------------------------------------------------------------
# 3. Web search (Perplexity Sonar through Kaya)
# ------------------------------------------------------------------
SEARCH_SYSTEM_PROMPT = (
    "Search the web and report the facts that answer the user's question. "
    "Be factual and concise. Include dates, numbers and names where relevant. "
    "Mark claims with citation numbers like 1, 2."
)


@lru_cache(maxsize=1)
def _search_client() -> openai.OpenAI:
    client = openai.OpenAI(
        api_key=os.environ["KAYA_API_KEY"],
        base_url=KAYA_BASE_URL,
        http_client=KAYA_HTTP_CLIENT,
        timeout=SEARCH_TIMEOUT,
        max_retries=MAX_RETRIES,
    )
    return wrap_openai(client)


def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _title_for(url: str, title: Optional[str]) -> str:
    return (title or "").strip() or urlparse(url).netloc or url


def _extract_sources(resp) -> List[Source]:
    """Read citations in all three known formats, de-duplicated by URL."""
    found = {}
    for item in _get(resp, "search_results") or []:
        url = _get(item, "url")
        if url:
            found.setdefault(url, _title_for(url, _get(item, "title")))
    for url in _get(resp, "citations") or []:
        if isinstance(url, str) and url:
            found.setdefault(url, _title_for(url, None))
    message = resp.choices[0].message if resp.choices else None
    for ann in _get(message, "annotations") or []:
        cit = _get(ann, "url_citation")
        url = _get(cit, "url") if cit else None
        if url:
            found.setdefault(url, _title_for(url, _get(cit, "title")))
    return [Source(title=title, url=url) for url, title in found.items()]


@traceable(name="web_search", run_type="retriever")
def web_search(query: str) -> Optional[SearchResult]:
    """Try each search model in order. None = search unavailable (not fatal)."""
    for model in SEARCH_MODELS:
        started = time.perf_counter()
        if DEBUG:
            print(f"   🌐 searching the web with {model} ...", flush=True)
        try:
            resp = _search_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=1200,
            )
        except openai.APIStatusError as e:
            if e.status_code in (401, 402):
                raise
            if DEBUG:
                print(f"   ❌ search failed ({model}): {type(e).__name__} {e.status_code}", flush=True)
            continue
        except openai.APIError as e:
            if DEBUG:
                print(f"   ❌ search failed ({model}): {type(e).__name__}", flush=True)
            continue

        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        sources = _extract_sources(resp)[:MAX_SOURCES]
        if DEBUG:
            print(f"   ✅ search done in {time.perf_counter() - started:.1f}s | {len(sources)} sources", flush=True)
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
# 4. Prompts
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise question-answering assistant. Today's date is {today}.
Rules:
- Answer directly and concisely.
- Use the earlier conversation to understand follow-up questions (words like "it", "that",
  "and in euros?"). For facts, the WEB RESULTS of the current question take priority over
  anything said earlier.
- If WEB RESULTS are provided, base your answer on them first. Keep citation markers like
  1, 2 that match the SOURCES list. If they don't answer the question, say so.
- If no WEB RESULTS are provided, answer from your own knowledge and LOWER the confidence
  for anything that may have changed recently.
- confidence scale: 0.9-1.0 well-established fact | 0.6-0.8 likely, some uncertainty |
  0.3-0.5 partial/uncertain | below 0.3 mostly guessing.
- If the question is ambiguous or unanswerable, say so and LOWER the confidence.
- sources: when WEB RESULTS are provided, return an empty list (real URLs are attached
  automatically). Otherwise cite only real, well-known references and NEVER invent URLs.
- Reply in the same language as the question."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history", optional=True),   # past Q&A goes here
    ("human", "Question: {question}\n\n{context}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You turn a follow-up question into ONE standalone web search query. "
     "Use the conversation to resolve references like 'it', 'that', 'and in euros?'. "
     "Write the query in the language most likely to find good sources "
     "(for example Persian for Iran-specific topics like the free-market rate). "
     "Return ONLY the query text, nothing else."),
    MessagesPlaceholder("history"),
    ("human", "Follow-up question: {question}"),
])


# ------------------------------------------------------------------
# 5. Models + chains
# ------------------------------------------------------------------
def get_model(name: str,
              temperature: float = 0.2,
              max_tokens: int = 1500,
              timeout: int = REQUEST_TIMEOUT,
              max_retries: int = MAX_RETRIES):
    return init_chat_model(
        model=name,
        model_provider="openai",
        api_key=os.environ["KAYA_API_KEY"],
        base_url=KAYA_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        http_client=KAYA_HTTP_CLIENT,
    )


def build_chain():
    structured_models = [
        get_model(name).with_structured_output(QAResponse, method="function_calling", include_raw=True)
        for name in MODEL_CANDIDATES
    ]
    return prompt | structured_models[0].with_fallbacks(structured_models[1:])


def build_rewriter():
    """Small, cheap call: follow-up question -> standalone search query (plain text)."""
    models = [get_model(name, temperature=0.0, max_tokens=100) for name in MODEL_CANDIDATES]
    return REWRITE_PROMPT | models[0].with_fallbacks(models[1:]) | StrOutputParser()


# ------------------------------------------------------------------
# 6. Live step printer (debug)
# ------------------------------------------------------------------
class StepPrinter(BaseCallbackHandler):
    def __init__(self):
        self._start = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        meta = kwargs.get("metadata") or {}
        params = kwargs.get("invocation_params") or {}
        name = params.get("model") or params.get("model_name") or meta.get("ls_model_name", "model")
        self._start[run_id] = time.perf_counter()
        print(f"   ⏳ calling {name} ...", flush=True)

    def on_llm_end(self, response, *, run_id, **kwargs):
        secs = time.perf_counter() - self._start.pop(run_id, time.perf_counter())
        try:
            msg = response.generations[0][0].message
            usage = getattr(msg, "usage_metadata", None) or {}
            tool = "tool call ✅" if getattr(msg, "tool_calls", None) else "text"
            tokens = usage.get("total_tokens", "?")
        except Exception:
            tool, tokens = "?", "?"
        print(f"   ✅ reply in {secs:.1f}s | {tool} | tokens: {tokens}", flush=True)

    def on_llm_error(self, error, *, run_id, **kwargs):
        secs = time.perf_counter() - self._start.pop(run_id, time.perf_counter())
        print(f"   ❌ failed after {secs:.1f}s: {type(error).__name__}: {error}", flush=True)


STEP_PRINTER = StepPrinter()


# ------------------------------------------------------------------
# 7. Core logic
# ------------------------------------------------------------------
def _message_text(message) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content).strip()


def make_search_query(rewriter, question: str, history: List[BaseMessage], config: dict) -> str:
    """No history -> the question as-is. With history -> a standalone query. Never fatal."""
    if not history:
        return question
    try:
        query = rewriter.invoke(
            {"question": question, "history": history},
            config={**config, "run_name": "rewrite_query"},
        )
    except openai.APIStatusError as e:
        if e.status_code in (401, 402):
            raise
        log.warning("Query rewrite failed (%s), using the raw question", e.status_code)
        return question
    except openai.APIError as e:
        log.warning("Query rewrite failed (%s), using the raw question", type(e).__name__)
        return question
    query = (query or "").strip().strip('"').strip()
    return query[:300] or question


@traceable(
    name="qa_turn",
    run_type="chain",
    tags=["qa-bot", "terminal"],
    process_inputs=lambda inputs: {
        "question": inputs.get("question"),
        "history_turns": len(inputs.get("history") or []) // 2,
    },
)
def ask(chain, rewriter, question: str, history: List[BaseMessage]) -> QAResponse:
    config = {"run_name": "qa_chain"}
    if DEBUG:
        config["callbacks"] = [STEP_PRINTER]

    # Step 1: search (follow-ups are rewritten first so the search makes sense)
    search = None
    if WEB_SEARCH:
        query = make_search_query(rewriter, question, history, config)
        if DEBUG and query != question:
            print(f"   🔁 search query: {query}", flush=True)
        search = web_search(query)
        if search is None and DEBUG:
            print("   ⚠️  web search unavailable, answering from memory", flush=True)

    # Step 2: structured answer, with the conversation history in the prompt
    inputs = {
        "question": question,
        "history": history,
        "context": build_context(search),
        "today": date.today().isoformat(),
    }
    resp = None
    result = None
    for attempt in range(1, PARSE_ATTEMPTS + 1):
        result = chain.invoke(inputs, config=config)
        if result.get("parsed") is not None:
            resp = result["parsed"]
            break
        log.warning("Structured parse failed (attempt %d): %s", attempt, result.get("parsing_error"))

    if resp is None:
        raw_text = _message_text(result.get("raw")) if result else ""
        fallback = raw_text or (search.text if search else "")
        resp = QAResponse(
            answer=fallback or "I couldn't produce a structured answer for that question.",
            confidence=0.0,
        )

    # Step 3: real URLs from the search + which models actually did the work
    if search and search.sources:
        resp.sources = search.sources
    raw = result.get("raw") if result else None
    meta = getattr(raw, "response_metadata", None) or {}
    resp._answered_by = meta.get("model_name") or "unknown"
    resp._searched_with = search.model if search else ""
    return resp


def safe_ask(chain, rewriter, question: str, memory: ConversationMemory,
             session_id: str) -> Tuple[Optional[QAResponse], Optional[str]]:
    extra = {"metadata": {"session_id": session_id}}
    try:
        return ask(chain, rewriter, question, memory.messages, langsmith_extra=extra), None
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
# 8. Terminal UI
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
    print(f"🤖 Answered by: {resp._answered_by}"
          + (f" | 🌐 searched with: {resp._searched_with}" if resp._searched_with else ""))
    print("-" * 60, flush=True)


def show_history(memory: ConversationMemory) -> None:
    if not len(memory):
        print("🧠 Memory is empty.\n")
        return
    print(f"🧠 Memory ({len(memory)}/{memory.max_turns} turns):")
    for msg in memory.messages:
        who = "You" if isinstance(msg, HumanMessage) else "Bot"
        text = str(msg.content).replace("\n", " ")
        print(f"   {who}: {text[:100]}{'...' if len(text) > 100 else ''}")
    print()


def main():
    if not os.getenv("KAYA_API_KEY"):
        sys.exit("❌ KAYA_API_KEY is missing from .env")

    tracing_on = _env_bool("LANGSMITH_TRACING", False)
    project = os.getenv("LANGSMITH_PROJECT", "default")
    print(f"🔍 LangSmith tracing: {'ON → project ' + project if tracing_on else 'OFF'}")
    print(f"🌐 Web search: {'ON → ' + ', '.join(SEARCH_MODELS) if WEB_SEARCH else 'OFF'}")
    print(f"🧠 Memory: {'last ' + str(MEMORY_TURNS) + ' turns' if MEMORY_TURNS else 'OFF'}")
    if DEBUG:
        print(f"🛠  Debug: ON | timeout={REQUEST_TIMEOUT}s | search timeout={SEARCH_TIMEOUT}s | retries={MAX_RETRIES}")
        print(f"🤖 Answer models (in order): {', '.join(MODEL_CANDIDATES)}")

    chain = build_chain()
    rewriter = build_rewriter()
    memory = ConversationMemory(MEMORY_TURNS)
    session_id = str(uuid.uuid4())
    print("💬 Ask me anything. Commands: /history, /reset, exit\n", flush=True)

    try:
        while True:
            try:
                raw = input("❓ You: ")
            except EOFError:
                print()
                break

            command = raw.strip().lower()
            if command in {"exit", "quit", "q"}:
                break
            if command == "/reset":
                memory.clear()
                print("🧹 Memory cleared.\n")
                continue
            if command == "/history":
                show_history(memory)
                continue

            question, err = validate(raw)
            if err:
                print(f"⚠️  {err}\n", flush=True)
                continue

            print("🤔 Thinking...", flush=True)
            started = time.perf_counter()
            resp, err = safe_ask(chain, rewriter, question, memory, session_id)
            if DEBUG:
                print(f"   ⏱  total: {time.perf_counter() - started:.1f}s", flush=True)

            if err:
                print(f"⚠️  {err}\n", flush=True)
                continue
            render(resp)
            memory.add(question, resp.answer)   # only successful turns are remembered
    except KeyboardInterrupt:
        print()
    finally:
        if tracing_on:
            print("📤 Sending remaining traces to LangSmith...", flush=True)
        wait_for_all_tracers()
        print("👋 Bye!")


if __name__ == "__main__":
    main()