"""Safety layer. Runs before retrieval and generation.

If a message suggests suicidal thoughts or self-harm, the assistant does not produce a normal RAG
answer. It shows a supportive message and real Saudi support numbers instead.

Detection is rule-based on normalized Arabic (MSA and dialect phrasings). It is tuned for recall:
showing support numbers to someone who did not need them costs little; missing a crisis costs a lot.
Numbers (verified October 2026, TherapyRoute directory of Saudi crisis lines):
    937          Ministry of Health call centre, 24/7 medical consultation
    920033360    Psychological Consultation Centre (MOH), mental-health specialists, until 8 pm
    997          Red Crescent ambulance (emergency)
    1919         Domestic violence reporting, 24/7
"""
from __future__ import annotations

import re

from .arabic import normalize

# Phrases are written normalized (أ/إ/آ -> ا, ة -> ه, ى -> ي, no diacritics).
_CRISIS = [
    r"انتحار", r"انتحر", r"اقتل نفسي", r"قتل نفسي", r"(?<!\S)[بوفل]?ء?(?:اذي|اوذي) نفسي", r"اذيت نفسي", r"اوذيت نفسي", r"يوذي نفسه", r"ايذاء النفس", r"ايذاء نفسي",
    r"اجرح نفسي", r"جرحت نفسي", r"اقطع (في )?(يدي|جسمي|عروقي)", r"انهي حياتي", r"انهاء حياتي", r"اخلص من حياتي",
    r"ما (ابي|ابغى|ابغي|اريد) اعيش", r"لا اريد ان اعيش", r"ما عاد (ابي|ابغى|ابغي) اعيش", r"مابي اعيش",
    r"اتمنى الموت", r"اتمني الموت", r"الموت ارحم", r"افكر (في|ب)(ال)?موت", r"افكار انتحاريه", r"ارمي نفسي",
    r"ابي اموت", r"ابغى اموت", r"اريد ان اموت", r"نفسي اموت", r"حياتي مالها (معنى|معني|داعي)",
]
_PATTERN = re.compile("|".join(f"(?:{p})" for p in _CRISIS))

SUPPORT_LINES = [
    ("937", "مركز اتصال وزارة الصحة، على مدار الساعة"),
    ("920033360", "مركز الاستشارات النفسية بوزارة الصحة، مختصون نفسيون، حتى الساعة 8 مساءً"),
    ("997", "الإسعاف، في حالات الخطر العاجل"),
    ("1919", "بلاغات العنف الأسري، على مدار الساعة"),
]

CRISIS_MESSAGE = ("يبدو أنك تمر بوقت صعب جداً، وشكراً لأنك كتبت عن ذلك. مشاعرك مهمة، ولست وحدك. "
                  "أنا مساعد آلي ولا أستطيع تقديم المساعدة التي تحتاجها الآن، لكن هناك أشخاص مختصون يستطيعون. "
                  "إذا كنت في خطر الآن أو تفكر في إيذاء نفسك، اتصل بالإسعاف 997 فوراً، أو تحدث مع شخص تثق به ليبقى معك.")


def is_crisis(text: str) -> bool:
    return bool(_PATTERN.search(normalize(text)))


def matched_phrase(text: str) -> str:
    m = _PATTERN.search(normalize(text))
    return m.group(0) if m else ""
