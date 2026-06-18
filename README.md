# Prompt-Based Data Labeling with Active Learning Loop

An intelligent system that uses LLM prompts to label datasets with high confidence by detecting systematic model bias through cross-model disagreement, prioritizing only genuinely uncertain rows for human review.

## Problem

Single-model self-consistency catches when a model wavers between samples, but **misses systematic bias**—when a model confidently gives the same wrong answer every time. Result: auto-confident predictions can still be incorrect.

## Solution

Sample **two independent models** and pool their votes. When they disagree on the majority label, the pooled distribution becomes uncertain—even if each model was individually confident.

**Example**: 
- Model A says "positive" (3/3 votes) for "Arrived on time, no complaints"
- Model B says "neutral" (3/3 votes) for the same text
- Pooled result: 50/50 split → uncertainty = 1.0 → flagged for human review

## Results (45-row test)

- **86.7%** auto-labeled (zero review needed)
- **6 rows** routed to human review
- **5/5** cross-model disagreements caught
- **3/3** systematic errors detected

## Architecture

**Input CSV** → **Model A (70B) + Model B (8B)** [k=3 self-consistency each] → **Uncertainty Estimation** (entropy + cross-model disagreement) → **Labeled Candidates** (sorted by uncertainty) → **Human Review** (high uncertainty only) → **Final Labels** → **Evaluate & Retrain**

## Key Files

| File | Purpose |
|------|---------|
| `scripts/label_batch.py` | Batch labeling with cross-model sampling |
| `ui/annotator_app.py` | Streamlit UI for human review |
| `core/uncertainty.py` | Shannon entropy + disagreement calculation |
| `scripts/evaluate.py` | F1 score vs gold labels |

## Why Cross-Model Disagreement?

Self-consistency entropy: **Stochastic uncertainty only** (model wavers between samples)

Cross-model pooling: **Catches systematic bias** (model confidently wrong due to training bias)

→ **Both signals combined** = complete uncertainty picture

## Technical Details

- **Self-consistency**: k=3 samples per model, temperature=0.7
- **Models**: llama-3.3-70b-versatile (Model A), llama-3.1-8b-instant (Model B)
- **Uncertainty metric**: Normalized Shannon entropy of pooled votes
- **Database**: SQLite for run tracking (cost, disagreements, metrics)

## Project Structure

├── core/
│   ├── llm_client.py            # Groq API wrapper
│   ├── uncertainty.py           # Entropy + disagreement
│   └── prompt_templates.py     # Classification prompts

├── scripts/
│   ├── label_batch.py          # Main labeling pipeline
│   ├── evaluate.py             # F1 vs gold labels
│   └── retrain.py              # Fine-tune on corrected data

├── ui/
│   └── annotator_app.py        # Streamlit review interface

├── data/
│   ├── input.csv
│   ├── labeled_candidates.csv
│   ├── final_labels.csv
│   └── gold.csv

└── tests/
    └── test_uncertainty.py

## Limitations & Future Work

- **Same-family bias**: Both models are Llama-based; may share correlated biases
- **Future**: Test with qwen/qwen3-32b (cross-family) or domain-specific fine-tuned models
- **Prompt optimization**: Could use DSPy for automated prompt tuning
- **Async sampling**: Current impl is sequential; could parallelize across batches

## License

MIT