"""
Terminal Q&A bot
question -> ChatPromptTemplate -> model.with_structured_output(QAResponse) -> QAResponse

- Structured output: answer + confidence + sources
- Resilience: per-model retries, cross-model fallbacks, parse retry, raw-text rescue
- Graceful failure: every exception becomes a readable message, the loop never dies
- LangSmith tracing: controlled only by env vars (.env)
- Debug mode: live step output in the terminal (QA_DEBUG=true)
"""
import os
import sys
import time
import uuid
import logging
from typing import List, Optional, Tuple
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

load_dotenv()


# ------------------------------------------------------------------
# Env helpers
# ------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.split("#")[0].strip())
    except ValueError:
        return default


# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
KAYA_BASE_URL = "https://kayaai.ir/api"
MAX_QUESTION_CHARS = 2000
PARSE_ATTEMPTS = 2  
# Kaya must never go through a proxy/VPN (it rejects foreign IPs)
KAYA_HTTP_CLIENT = httpx.Client(trust_env=False)# 1 normal try + 1 retry on bad structure
MODEL_CANDIDATES = [
    "deepseek/deepseek-v4-pro-0813",        # primary
    "openai/gpt-5.6-luna-pro",              # fallback 1
    "meta/muse-spark-1.3-contributor",      # fallback 2
]

DEBUG = _env_bool("QA_DEBUG", True)         # live step output
VERBOSE = _env_bool("QA_VERBOSE", False)    # full LangChain debug dump (very long)
REQUEST_TIMEOUT = _env_int("QA_TIMEOUT", 60)
MAX_RETRIES = _env_int("QA_MAX_RETRIES", 2)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("qa_bot")

if DEBUG:
    logging.getLogger("httpx").setLevel(logging.INFO)    # every HTTP request + status code
    logging.getLogger("openai").setLevel(logging.INFO)   # "Retrying request..." lines

if VERBOSE:
    set_debug(True)                         # prints every prompt, message and raw reply


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
    sources: List[Source] = Field(default_factory=list, description="Well-known references supporting the answer. Empty list if none")


# ------------------------------------------------------------------
# 2. Prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a precise question-answering assistant.
Rules:
- Answer directly and concisely.
- confidence scale: 0.9-1.0 well-established fact | 0.6-0.8 likely, some uncertainty |
  0.3-0.5 partial/uncertain | below 0.3 mostly guessing.
- If the question is ambiguous, unanswerable, or about events after your knowledge cutoff,
  say so in the answer and LOWER the confidence.
- sources: cite only real, well-known references (official docs, textbooks, encyclopedias).
  NEVER invent URLs. Use null for url if unsure.
- Reply in the same language as the question."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ------------------------------------------------------------------
# 3. Model + chain
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
        http_client=KAYA_HTTP_CLIENT,
    )


def build_chain():
    """
    Bind the schema to EACH model, then chain them with fallbacks,
    so every path returns the same {raw, parsed, parsing_error} shape.
    """
    structured_models = [
        get_model(name).with_structured_output(
            QAResponse,
            method="function_calling",      # most compatible on OpenAI-style proxies
            include_raw=True,               # bad structure -> no exception, just parsed=None
        )
        for name in MODEL_CANDIDATES
    ]
    llm = structured_models[0].with_fallbacks(structured_models[1:])
    return prompt | llm


# ------------------------------------------------------------------
# 4. Live step printer (debug)
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
        print(f"   ⏳ calling {name} ...", flush=True)

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
# 5. Core logic (one traced "turn" in LangSmith)
# ------------------------------------------------------------------
def _message_text(message) -> str:
    """Safely extract plain text from an AIMessage (content may be str, list, or missing)."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):           # some providers return content blocks
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content).strip()


@traceable(
    name="qa_turn",
    run_type="chain",
    tags=["qa-bot", "terminal"],
    process_inputs=lambda inputs: {"question": inputs.get("question")},  # hide the chain object
)
def ask(chain, question: str) -> QAResponse:
    config = {"run_name": "qa_chain"}
    if DEBUG:
        config["callbacks"] = [STEP_PRINTER]   # passed down to every model call in the chain
    result = None

    for attempt in range(1, PARSE_ATTEMPTS + 1):
        result = chain.invoke({"question": question}, config=config)
        if result.get("parsed") is not None:
            return result["parsed"]
        log.warning("Structured parse failed (attempt %d): %s", attempt, result.get("parsing_error"))

    # Salvage: the model replied, but not in the schema
    raw_text = _message_text(result.get("raw")) if result else ""
    return QAResponse(
        answer=raw_text or "I couldn't produce a structured answer for that question.",
        confidence=0.0,
        sources=[],
    )


def safe_ask(chain, question: str, session_id: str) -> Tuple[Optional[QAResponse], Optional[str]]:
    """Graceful failure layer: turns every exception into a readable message."""
    extra = {"metadata": {"session_id": session_id}}   # groups turns into a Thread in LangSmith
    try:
        return ask(chain, question, langsmith_extra=extra), None
    except openai.AuthenticationError:
        return None, "Authentication failed. Check KAYA_API_KEY."
    except openai.RateLimitError:
        return None, "Rate limit hit on all models. Wait a bit and try again."
    except (openai.APITimeoutError, openai.APIConnectionError):
        return None, "Network/timeout problem reaching the API. Check your connection (try VPN off for Kaya)."
    except openai.APIStatusError as e:
        return None, f"Provider returned an error ({e.status_code}). Try again later."
    except Exception as e:                  # last-resort safety net
        log.exception("Unexpected error")
        return None, f"Unexpected error: {type(e).__name__}"


# ------------------------------------------------------------------
# 6. Terminal UI
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
            print(f"   {i}. {s.title}" + (f" | {s.url}" if s.url else ""))
    else:
        print("📚 Sources: none cited")
    print("-" * 60, flush=True)


def main():
    if not os.getenv("KAYA_API_KEY"):
        sys.exit("❌ KAYA_API_KEY is missing from .env")

    tracing_on = _env_bool("LANGSMITH_TRACING", False)
    project = os.getenv("LANGSMITH_PROJECT", "default")
    print(f"🔍 LangSmith tracing: {'ON → project ' + project if tracing_on else 'OFF'}")
    if DEBUG:
        print(f"🛠  Debug: ON | timeout={REQUEST_TIMEOUT}s | retries={MAX_RETRIES} | verbose={VERBOSE}")
        print(f"🧠 Models (in order): {', '.join(MODEL_CANDIDATES)}")

    chain = build_chain()
    session_id = str(uuid.uuid4())
    print("🤖 Ask me anything. Type 'exit' to quit.\n", flush=True)

    try:
        while True:
            try:
                raw = input("❓ You: ")
            except EOFError:                # Ctrl+D / Ctrl+Z+Enter / end of piped input
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
    except KeyboardInterrupt:               # Ctrl+C, even during a model call
        print()
    finally:
        if tracing_on:
            print("📤 Sending remaining traces to LangSmith...", flush=True)
        wait_for_all_tracers()              # flush pending traces before exit
        print("👋 Bye!")


if __name__ == "__main__":
    main()