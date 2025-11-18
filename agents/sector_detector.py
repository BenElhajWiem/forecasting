from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SectorConfig:
    sectors: List[str] = (
        "Retail","Finance","Healthcare",
        "Manufacturing","Nature","Energy","Transportation"
    )
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 200
    fallback: str = "Undetected"   # used if the model gives something invalid


class SectorDetector:
    def __init__(self, cfg: SectorConfig = SectorConfig()):
        self.cfg = cfg

    def classify(self, adapter, query: str) -> str:
        """
        Classify a query into exactly one sector word.
        """
        # 1) Build prompt
        system =(
            "You are a a deterministic multiclass classifier.\n"
            f"Given a query output EXACTLY one word from the allowed list: {', '.join(self.cfg.sectors)}.\n"
            "Rules:\n"
            "1) No punctuation, no quotes, no explanations, no JSON.\n"
            "2) If uncertain, choose the single best fit; if nothing fits, output Undetected.\n"
            "### FEW-SHOT EXAMPLES ###\n"
            "Query: Predict electricity demand in NSW.\n"
            "Answer: Energy\n\n"
            "Query: Summarize recent stock price trends for Apple.\n"
            "Answer: Finance\n\n"
            "Query: Generate a meal plan for a high-protein diet.\n"
            "Answer: Health\n\n"
            "### END OF EXAMPLES ###\n"
        )
        user = (
            "Classify the following query into exactly ONE category from the allowed list.\n"
            f"Query: {query}\n"
            "Answer with only the single category word."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # 2) Call the LLM
        raw = adapter.chat(
            messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            model_override=self.cfg.model,
        )

        # 3) Normalize
        out = str(raw).strip()
        for s in self.cfg.sectors:
            if out.lower() == s.lower():
                return s

        # 4) Fallback
        return self.cfg.fallback