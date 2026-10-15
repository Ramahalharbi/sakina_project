"""Answer generation with OpenAI GPT (default gpt-4o-mini).

GPT only sees the specialist answers retrieved for the person's message, and is instructed to be
supportive, never diagnose, never name or dose medicines, and always recommend a specialist.
Without an API key (or if the request fails), the fallback returns the specialist answer of the most
similar real case, clearly labelled as such.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from . import config
from .retrieval import Hit

PROMPT = """أنت "سكينة"، مساعد دعم نفسي باللغة العربية. لست طبيباً ولا تشخّص.
ستجد أدناه حالات حقيقية مشابهة أجاب عنها مختصون في الصحة النفسية. اعتمد على إجاباتهم فقط.
القواعد:
- ابدأ بجملة تعاطف قصيرة وصادقة.
- اشرح بلغة بسيطة ما قاله المختصون في الحالات المشابهة، دون إضافة معلومات من عندك.
- لا تذكر أسماء أدوية ولا جرعات، ولا تطلب إيقاف دواء أو تغييره؛ قل إن ذلك يكون مع الطبيب.
- لا تشخّص الحالة.
- اختم دائماً بتشجيع مراجعة مختص نفسي.
- إذا كانت الحالات لا تتعلق بسؤال الشخص فقل ذلك بلطف ووجّهه لمختص.
أعد JSON فقط بالشكل:
{{"answer": "الرد في 3 إلى 5 جمل", "steps": ["خطوة عملية", "خطوة عملية"], "sources": [1, 2]}}

الحالات المشابهة:
{context}

رسالة الشخص: {question}"""

ALWAYS_NOTE = "هذا الرد للدعم والتوعية فقط ولا يغني عن مختص نفسي. إذا استمرت الأعراض أو أثرت على حياتك اليومية فراجع طبيباً أو أخصائياً نفسياً."


@dataclass
class Reply:
    answer: str
    steps: list[str]
    sources: list[int]
    mode: str                                  # llm | retrieved
    hits: list[Hit] = field(default_factory=list)
    generation_ms: float = 0.0
    error: str = ""                            # why GPT was not used (empty when GPT replied)


def llm_available() -> bool:
    return bool(config.openai_api_key())


def _gpt(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=config.openai_api_key())
    resp = client.chat.completions.create(
        model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def _context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] سؤال: {h.chunk.case.question}\nإجابة المختص: {h.chunk.case.answer}"
                       for i, h in enumerate(hits, 1))


def generate(question: str, hits: list[Hit], use_llm: bool = True) -> Reply:
    t0 = time.perf_counter()
    reply, error = None, ""
    if use_llm and hits and not llm_available():
        error = "لم يُعثر على مفتاح OpenAI. ضع الملف secrets.toml داخل مجلد .streamlit في مجلد المشروع."
    if use_llm and hits and llm_available():
        try:
            raw = _gpt(PROMPT.format(context=_context(hits), question=question))
            data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            srcs = [int(s) for s in data.get("sources", []) if str(s).isdigit() and 1 <= int(s) <= len(hits)]
            reply = Reply(str(data.get("answer", "")).strip(), [str(s) for s in data.get("steps", [])][:4], srcs, "llm", hits)
        except Exception as exc:
            reply, error = None, config.explain_gpt_error(exc)
    if reply is None or not reply.answer:
        top = hits[0].chunk.case.answer if hits else ""
        reply = Reply(top, [], [1] if hits else [], "retrieved", hits, error=error or "رد GPT فارغ.")
    reply.generation_ms = (time.perf_counter() - t0) * 1000
    return reply
