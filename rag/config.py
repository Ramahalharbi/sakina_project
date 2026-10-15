"""Configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = DATA_DIR / "results"

# How each knowledge-base case is stored and searched ("unit"). Compared in the evaluation:
#   qa        question + specialist answer in one unit (tied with "question" on dev)
#   question  search the patient question only, return its answer (default, chosen on dev)
#   answer    search the specialist answer only
UNIT = os.getenv("UNIT", "question")      # chosen on dev (leave-one-out) by evaluation.choose_unit

# Embeddings (free, run locally). Falls back to offline LSA if the model cannot be downloaded.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

# Retrieval
RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "hybrid")     # hybrid | dense | bm25 | tfidf
TOP_K = int(os.getenv("TOP_K", 3))
RRF_K = 60

# Large language model: OpenAI GPT. The key is read from the OPENAI_API_KEY environment variable or from
# .streamlit/secrets.toml (OPENAI_API_KEY = "sk-..."); it is never stored in the code.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))


def openai_api_key() -> str:
    """Key lookup order: OPENAI_API_KEY env var, Streamlit secrets, then the project's own
    .streamlit/secrets.toml read directly (works even if streamlit is started from another folder)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        if key:
            return key
    except Exception:
        pass
    path = ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        try:
            import tomllib

            return str(tomllib.loads(path.read_text(encoding="utf-8")).get("OPENAI_API_KEY", "")).strip()
        except Exception:
            return ""
    return ""


def explain_gpt_error(exc: Exception) -> str:
    """Short Arabic explanation of why the GPT call failed, for the person running the app."""
    name, text = type(exc).__name__, str(exc)
    if "AuthenticationError" in name or "invalid_api_key" in text or "Incorrect API key" in text:
        return "المفتاح غير صحيح. تأكد أنك نسخته كاملاً في .streamlit/secrets.toml."
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return "لا يوجد رصيد في حساب OpenAI. أضف رصيداً من صفحة Billing ثم أعد المحاولة."
    if "RateLimitError" in name:
        return "تجاوزت حد الطلبات مؤقتاً. انتظر قليلاً ثم أعد المحاولة."
    if "model_not_found" in text or "does not exist" in text:
        return f"النموذج {LLM_MODEL} غير متاح لحسابك."
    if "APIConnectionError" in name or "Timeout" in name:
        return "تعذر الاتصال بخدمة OpenAI. تحقق من اتصال الإنترنت."
    if "ModuleNotFoundError" in name:
        return "مكتبة openai غير مثبتة. شغّل: pip install openai"
    return f"تعذر الاتصال بـ GPT ({name})."
