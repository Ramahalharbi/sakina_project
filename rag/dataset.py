"""MentalQA: An Annotated Arabic Corpus for Questions and Answers of Mental Healthcare.
Alhuzali, Alasmari, Alsaleh. IEEE Access 2024. MIT license. https://github.com/hasanhuz/MentalQA

Real data: questions written by people seeking mental-health help, each answered by a mental-health
professional, annotated with question types and answer strategies.

How the project uses the files (all in data/raw):
- Train_Dev.tsv (350 cases: question, answer, question types, answer strategies)
      -> the knowledge base the assistant retrieves from.
- Subtask3_input_test.tsv / Subtask3_output_test.tsv (150 new questions + specialist answers)
      -> test set for answer generation. 3 questions that also appear in the knowledge base are
         removed, so the assistant never sees a test answer.
- subtask1_input_test.tsv / subtask1_output_test.tsv (150 new questions + question-type labels)
      -> test set for question-type classification and retrieval relevance.
In the AraHealthQA 2025 release these test files contain different questions, so they are used separately.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from . import config

QUESTION_TYPES = {
    "A": "التشخيص", "B": "العلاج", "C": "التشريح ووظائف الجسم", "D": "أسباب المرض ومساره",
    "E": "نمط الحياة الصحي", "F": "اختيار مقدّم الرعاية", "Z": "أخرى",
}
ANSWER_STRATEGIES = {"1": "معلومات", "2": "توجيه مباشر", "3": "دعم عاطفي"}


@dataclass
class Case:
    case_id: str
    question: str
    answer: str
    types: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)


def _ws(t: str) -> str:
    return " ".join(str(t).split())


def _labels(v) -> list[str]:
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x).strip() for x in ast.literal_eval(s)]
        except (ValueError, SyntaxError):
            pass
    return [x.strip() for x in s.replace("[", "").replace("]", "").replace("'", "").split(",") if x.strip()]


def _lines(name: str) -> list[str]:
    return [l.rstrip("\r\n") for l in (config.RAW_DIR / name).read_text(encoding="utf-8").split("\n") if l.strip()]


@lru_cache(maxsize=1)
def knowledge_base() -> list[Case]:
    df = pd.read_csv(config.RAW_DIR / "Train_Dev.tsv", sep="\t")
    return [Case(f"kb{i:03d}", _ws(r.question), _ws(r.answer), _labels(r.final_QT), _labels(r.final_AS))
            for i, r in enumerate(df.itertuples()) if str(r.question).strip() and str(r.answer).strip()]


@lru_cache(maxsize=1)
def generation_test() -> list[Case]:
    kb_questions = {c.question for c in knowledge_base()}
    qs, ans = _lines("Subtask3_input_test.tsv"), _lines("Subtask3_output_test.tsv")
    return [Case(f"g{i:03d}", _ws(q), _ws(a)) for i, (q, a) in enumerate(zip(qs, ans)) if _ws(q) not in kb_questions]


@lru_cache(maxsize=1)
def classification_test() -> list[Case]:
    qs, labels = _lines("subtask1_input_test.tsv"), _lines("subtask1_output_test.tsv")
    return [Case(f"c{i:03d}", _ws(q), "", _labels(l)) for i, (q, l) in enumerate(zip(qs, labels))]


def stats() -> dict:
    kb = knowledge_base()
    types = pd.Series([t for c in kb for t in c.types]).value_counts().to_dict()
    strat = pd.Series([s for c in kb for s in c.strategies]).value_counts().to_dict()
    return {"knowledge_base": len(kb), "generation_test": len(generation_test()),
            "generation_test_removed_overlap": 150 - len(generation_test()),
            "classification_test": len(classification_test()),
            "avg_question_chars": int(sum(len(c.question) for c in kb) / len(kb)),
            "avg_answer_chars": int(sum(len(c.answer) for c in kb) / len(kb)),
            "type_counts": {QUESTION_TYPES.get(k, k): int(v) for k, v in types.items()},
            "strategy_counts": {ANSWER_STRATEGIES.get(k, k): int(v) for k, v in strat.items()}}
