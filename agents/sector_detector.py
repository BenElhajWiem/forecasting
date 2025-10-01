from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SectorConfig:
    sectors: List[str] = (
        "Retail",
        "Finance",
        "Healthcare",
        "Manufacturing",
        "Nature",
        "Energy",
        "Transportation",
    )
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 5
    fallback: str = "Energy"   # used if the model gives something invalid


class SectorDetector:
    def __init__(self, cfg: SectorConfig = SectorConfig()):
        self.cfg = cfg

    def classify(self, adapter, query: str) -> str:
        """
        Classify a query into exactly one sector word.

        Parameters
        ----------
        adapter : object
            Your LLM client with a .chat(messages, temperature, max_tokens, model_override) method.
        query : str
            The user's query.

        Returns
        -------
        str
            One of the sectors defined in cfg.sectors.
        """
        # 1) Build prompt
        system = (
            "Classify the user's query into exactly one sector. "
            "Reply with ONLY one word, chosen from this list:\n"
            f"{', '.join(self.cfg.sectors)}\n"
            "No punctuation. No sentences. No JSON. Only the single sector word."
        )
        user = f"Query: {query}"

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