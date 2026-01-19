"""
Builds a tiny sample dataset for local/Colab testing.
Run: python -m src.data.prepare
"""

from pathlib import Path
import pandas as pd
from src.config import PROCESSED_DIR

def main():
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    data = [
        {
            "context": "The Eiffel Tower is located in Paris.",
            "answer": "Paris",
            "question": "Where is the Eiffel Tower located?",
            "difficulty": "easy"
        },
        {
            "context": "Albert Einstein developed the theory of relativity.",
            "answer": "Albert Einstein",
            "question": "Who developed the theory of relativity?",
            "difficulty": "medium"
        }
    ]
    df = pd.DataFrame(data)
    df.to_csv(f"{PROCESSED_DIR}/sample.csv", index=False)
    print(f"Saved {len(df)} samples → {PROCESSED_DIR}/sample.csv")

if __name__ == "__main__":
    main()
