"""Generates a synthetic dataset for testing the labeling pipeline: short
product/app review snippets with sentiment labels. Run once before
label_batch.py to produce data/input.csv and data/gold.csv.
"""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# (text, true_label) — true_label is only used to build the gold subset;
# the labeling pipeline itself never sees it until evaluation.
SAMPLES = [
    ("Battery lasts barely 4 hours, way short of the advertised 10.", "negative"),
    ("Setup took two minutes and it just worked. Genuinely impressed.", "positive"),
    ("It's fine. Does what it says, nothing more.", "neutral"),
    ("Customer support ignored three emails before I gave up.", "negative"),
    ("The camera quality on this is shockingly good for the price.", "positive"),
    ("Arrived on time, packaging was okay, no complaints really.", "neutral"),
    ("Crashed twice during setup, had to factory reset.", "negative"),
    ("Best purchase I've made this year, recommend to everyone.", "positive"),
    ("It works as described, but the app UI feels dated.", "neutral"),
    ("Returned it after a week, build quality felt cheap.", "negative"),
    ("Five stars, exceeded what I expected at this price point.", "positive"),
    ("Average product, similar to others in this price range.", "neutral"),
    ("Stopped charging after two weeks, total waste of money.", "negative"),
    ("Sound quality is incredible, bass is deep without distortion.", "positive"),
    ("Instructions were unclear but I figured it out eventually.", "neutral"),
    ("Customer service resolved my issue within an hour, great experience.", "positive"),
    ("The fabric started fraying after just two washes.", "negative"),
    ("It's okay for the price, wouldn't pay more than this.", "neutral"),
    ("Completely broke on day one, requesting a refund.", "negative"),
    ("Love how lightweight it is, perfect for travel.", "positive"),
    ("Delivery took three weeks longer than promised.", "negative"),
    ("Does the job, no surprises good or bad.", "neutral"),
    ("The interface is so intuitive my parents figured it out instantly.", "positive"),
    ("Smells strongly of chemicals straight out of the box.", "negative"),
    ("Decent enough, I'll probably buy the upgraded version next.", "neutral"),
    ("This app drains my battery faster than any other app I own.", "negative"),
    ("Incredible value, didn't expect this quality at this price.", "positive"),
    ("It's neither good nor bad, just exists.", "neutral"),
    ("My order arrived damaged and missing parts.", "negative"),
    ("The customer onboarding flow is the smoothest I've seen.", "positive"),
    ("Works fine on wifi but constantly drops on mobile data.", "negative"),
    ("Solid build, nothing flashy, gets the job done.", "neutral"),
    ("I've recommended this to three friends already, that good.", "positive"),
    ("The subscription auto-renewed without any warning email.", "negative"),
    ("Looks exactly like the photos, happy with the purchase.", "positive"),
    ("Mediocre, wouldn't actively recommend but wouldn't warn against it either.", "neutral"),
    ("Support chat just looped me through the same three questions.", "negative"),
    ("Genuinely the best headphones I've owned, worth every rupee.", "positive"),
    ("It's a standard option, comparable to competitors.", "neutral"),
    ("Screen cracked from a minor drop, fragile material.", "negative"),
    ("Setup wizard walked me through everything, zero confusion.", "positive"),
    ("Fine for casual use, wouldn't trust it for anything critical.", "neutral"),
    ("Refund process took over a month, very frustrating.", "negative"),
    ("Exceeded expectations, the build quality feels premium.", "positive"),
    ("Just an average experience overall, nothing memorable.", "neutral"),
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    rows = [{"id": i + 1, "text": text} for i, (text, _) in enumerate(SAMPLES)]

    with open(DATA_DIR / "input.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        writer.writeheader()
        writer.writerows(rows)

    # Gold subset: every 3rd row, with its true label, for evaluation.
    gold_rows = [
        {"id": i + 1, "text": text, "gold_label": label}
        for i, (text, label) in enumerate(SAMPLES) if i % 3 == 0
    ]
    with open(DATA_DIR / "gold.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "gold_label"])
        writer.writeheader()
        writer.writerows(gold_rows)

    print(f"Wrote {len(rows)} rows to data/input.csv")
    print(f"Wrote {len(gold_rows)} gold rows to data/gold.csv")


if __name__ == "__main__":
    main()