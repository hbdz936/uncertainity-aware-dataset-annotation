"""Thin wrapper around the Groq API for classification calls, with retries
and basic cost/token accounting.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.prompt_templates import build_classification_prompt

# Approximate Groq pricing per 1M tokens (USD) — illustrative for cost
# tracking, not billing-accurate. Update if Groq's pricing changes.
PRICING_PER_1M_TOKENS = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}
DEFAULT_MODEL = "llama-3.3-70b-versatile"


@dataclass
class ClassificationResult:
    label: str
    raw_response: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class GroqLabelClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def classify(self, text: str, labels: list[str], temperature: float = 0.7) -> ClassificationResult:
        prompt = build_classification_prompt(text, labels)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            temperature=temperature,
            max_tokens=60,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        label = parsed.get("label", "").strip()
        if label not in labels:
            label = self._closest_label(label, labels)

        usage = response.usage
        cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens)

        return ClassificationResult(
            label=label,
            raw_response=content,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = PRICING_PER_1M_TOKENS.get(self.model)
        if not pricing:
            return 0.0
        return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

    @staticmethod
    def _closest_label(predicted: str, labels: list[str]) -> str:
        predicted_lower = predicted.lower()
        for label in labels:
            if label.lower() in predicted_lower or predicted_lower in label.lower():
                return label
        return labels[0]  # fallback