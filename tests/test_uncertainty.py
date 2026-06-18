import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.uncertainty import normalized_entropy, majority_vote, cross_model_disagreement


def test_unanimous_votes_zero_entropy():
    assert normalized_entropy(["positive", "positive", "positive"]) == 0.0


def test_max_disagreement_three_way_split():
    assert normalized_entropy(["positive", "negative", "neutral"]) == 1.0


def test_majority_vote_basic():
    label, confidence, uncertainty = majority_vote(["positive", "positive", "negative"])
    assert label == "positive"
    assert confidence == 2 / 3
    assert 0 < uncertainty < 1


def test_two_way_split_more_certain_than_three_way():
    two_way = normalized_entropy(["positive", "positive", "negative"])
    three_way = normalized_entropy(["positive", "negative", "neutral"])
    assert two_way < three_way


def test_cross_model_disagreement_flags_mismatch():
    assert cross_model_disagreement("positive", "neutral") == 1
    assert cross_model_disagreement("positive", "positive") == 0