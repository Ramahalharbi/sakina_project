"""Checks that the OpenAI key works, and prints the exact problem if it does not.

    python scripts/check_gpt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config  # noqa: E402

key = config.openai_api_key()
if not key:
    print("No key found.")
    print(f"Create {config.ROOT / '.streamlit' / 'secrets.toml'} containing:  OPENAI_API_KEY = \"sk-...\"")
    sys.exit(1)
print(f"Key found: {key[:7]}...{key[-4:]}  (model: {config.LLM_MODEL})")
try:
    from openai import OpenAI

    resp = OpenAI(api_key=key).chat.completions.create(
        model=config.LLM_MODEL, messages=[{"role": "user", "content": "قل: مرحباً"}], max_tokens=10)
    print("GPT works. Reply:", resp.choices[0].message.content)
except Exception as exc:
    print("GPT failed:", config.explain_gpt_error(exc))
    print("Details:", type(exc).__name__, str(exc)[:300])
    sys.exit(1)
