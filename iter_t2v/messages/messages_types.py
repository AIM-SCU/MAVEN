# /workspace/t2v_self/iter_t2v/messages/messages_types.py
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class Prompt:
    # ─── PAL metadata ───────────────────────────────────────────────────
    person_culture: str
    action_culture: str
    location_culture: str
    action_category: str
    alignment: str                 # "same_culture" | "mixed_culture"
    experiment_tag: str

    # ─── Prompt text + explicit segments ───────────────────────────────
    text: str                      # full sentence
    person_segment: str = ""       # e.g. "a Chinese person"
    action_segment: str = ""       # e.g. "eating Peking duck"
    location_segment: str = ""     # e.g. "the Forbidden City"

    # ------------------------------------------------------------------
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Prompt":
        return cls(
            person_culture=row["person_culture"],
            action_culture=row["action_culture"],
            location_culture=row["location_culture"],
            action_category=row["action_category"],
            alignment=row["alignment"],
            experiment_tag=row.get("experiment_tag", "unspecified"),
            text=row["prompt"],
            person_segment=row.get("person_segment", ""),
            action_segment=row.get("action_segment", ""),
            location_segment=row.get("location_segment", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # noqa: DunderStr
        return f"<Prompt {self.person_culture}/{self.action_category} | {self.text}>"
