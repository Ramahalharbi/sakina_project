"""Arabic text normalization and tokenization shared by indexing, retrieval and evaluation."""
import re

_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)

# Small, high-frequency stopword list (MSA + common Saudi dialect fillers).
STOPWORDS = set(
    """
    في من على الى إلى عن مع او أو ثم ان أن إن كان كانت يكون تكون هذا هذه ذلك تلك التي الذي الذين
    ما ماذا هل لا لم لن قد كل اي أي بعد قبل عند حتى اذا إذا بين كما لكن او و ف ب ل ك
    هو هي هم انا أنا انت أنت نحن له لها لهم به بها فيه فيها منه منها عليه عليها
    يا وش ايش إيش يبي ابي أبي ابغى أبغى مين وين كيف ليش متى
    """.split()
)


def normalize(text: str) -> str:
    """Normalize Arabic orthographic variation so retrieval and metrics are not fooled by spelling."""
    if not text:
        return ""
    text = text.translate(_ARABIC_DIGITS)
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = _NON_WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _light_stem(tok: str) -> str:
    """Strip the definite article and common attached prefixes; keeps roots comparable."""
    for p in ("وال", "بال", "كال", "فال", "لل", "ال"):
        if tok.startswith(p) and len(tok) - len(p) >= 2:
            return tok[len(p):]
    return tok


_STOP_N = {normalize(w) for w in STOPWORDS}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    toks = normalize(text).split()
    if remove_stopwords:
        toks = [t for t in toks if t not in _STOP_N]
    toks = [_light_stem(t) for t in toks]
    if remove_stopwords:
        toks = [t for t in toks if len(t) > 1]
    return toks
