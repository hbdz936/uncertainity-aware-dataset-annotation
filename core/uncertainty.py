"""Uncertainty estimation for LLM-labeled examples via self-consistency sampling
and cross-model disagreement.

Why entropy over a raw disagreement-rate: entropy captures *how* split the votes
are, not just whether they're unanimous. A 2-1 split across two labels is less
uncertain than a 1-1-1 split across three labels, even though neither has a
clean majority.

Self-consistency alone (resampling one model) only catches stochastic
uncertainty — cases where the model wavers between samples. It's blind to
systematic bias: a model that confidently gives the same wrong answer every
time. Pooling votes from a second, independent model surfaces that — if
model A says "positive" 3/3 and model B says "neutral" 3/3, the pooled vote
distribution is a 50/50 split (entropy 1.0), even though neither model
individually showed any internal doubt.
"""
from __future__ import annotations
from collections import Counter
from math import log2


def vote_distribution(labels: list[str]) -> Counter:
    return Counter(labels)


def normalized_entropy(labels: list[str]) -> float:
    """Shannon entropy of the vote distribution, normalized to [0, 1].

    0.0 = perfect agreement (every sample picked the same label)
    1.0 = maximum disagreement (votes spread evenly across all labels seen)
    """
    counts = vote_distribution(labels)
    n = len(labels)
    if n == 0 or len(counts) == 1:
        return 0.0

    probs = [c / n for c in counts.values()]
    raw_entropy = -sum(p * log2(p) for p in probs if p > 0)
    max_entropy = log2(len(counts))
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def majority_vote(labels: list[str]) -> tuple[str, float, float]:
    """Returns (majority_label, agreement_confidence, normalized_entropy)."""
    counts = vote_distribution(labels)
    majority_label, majority_count = counts.most_common(1)[0]
    confidence = majority_count / len(labels)
    uncertainty = normalized_entropy(labels)
    return majority_label, confidence, uncertainty


def cross_model_disagreement(label_a: str, label_b: str) -> int:
    """1 if two models' majority labels disagree, 0 if they agree.

    Kept as an explicit flag alongside pooled-vote entropy: entropy tells you
    uncertainty exists, this tells you whether it's coming from inter-model
    disagreement specifically (vs. one model wavering internally).
    """
    return int(label_a != label_b)