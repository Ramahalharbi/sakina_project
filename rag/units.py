"""What is stored and searched for each knowledge-base case (the "chunking" decision for short Q&A data).

MentalQA answers are short (about 180 characters), so splitting text into pieces is not the
question here; the question is what one stored unit should contain:
    qa        patient question + specialist answer together (default)
    question  the patient question only (match people with similar situations)
    answer    the specialist answer only (match the advice itself)
The evaluation compares the three. Whatever the unit, the assistant returns the full case.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dataset import Case

UNITS = ["qa", "question", "answer"]


@dataclass
class Unit:
    case: Case
    mode: str

    @property
    def chunk_id(self) -> str:
        return f"{self.case.case_id}:{self.mode}"

    @property
    def text(self) -> str:
        return self.case.answer

    @property
    def embed_text(self) -> str:
        if self.mode == "question":
            return self.case.question
        if self.mode == "answer":
            return self.case.answer
        return f"سؤال: {self.case.question}\nإجابة المختص: {self.case.answer}"


def make_units(cases: list[Case], mode: str) -> list[Unit]:
    return [Unit(c, mode) for c in cases]
