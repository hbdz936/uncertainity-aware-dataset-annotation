"""Prompt construction for the classification task.

Kept deliberately simple (single zero-shot template) so the active-learning
loop's value is visible independent of prompt sophistication. Swap in few-shot
examples or chain-of-thought here without touching the labeling pipeline.
"""

SYSTEM_PROMPT = (
    "You are a precise text classifier. Given a piece of text and a fixed set "
    "of labels, choose exactly one label that best fits. Respond ONLY with a "
    'JSON object: {"label": "<one of the given labels>"}. No explanation.'
)


def build_classification_prompt(text: str, labels: list[str]) -> list[dict]:
    labels_str = ", ".join(labels)
    user_prompt = f'Labels: [{labels_str}]\n\nText: """{text}"""\n\nJSON:'
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]