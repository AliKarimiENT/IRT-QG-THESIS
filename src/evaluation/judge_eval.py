import os
import re
import json
import time
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI SDK not found. Install with: pip install -U openai") from e

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

INPUT_FILE = "/kaggle/working/generated_samples.csv"
OUTPUT_FILE = "llm_judged_results.csv"

PREFERRED_MODELS = [
    os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4o",
]

RESPONSE_FORMAT = {"type": "json_object"}
MAX_WORKERS = 3

QUESTION_TYPES = [
    "readingMatchingFeatures", "readingMultipleChoices", "readingShortAnswer",
    "readingTextCompletion", "readingTrueFalseNotGiven", "readingYesNoNotGiven"
]
QUESTION_TYPES_JSON = json.dumps(QUESTION_TYPES)

PROMPT_TEMPLATE = """
You are an expert evaluator for an IELTS Question Generation system. Your task is to 
assess the quality of a generated question based on a given passage.
Return ONLY a JSON object with the required keys and nothing else.

**Passage:**
---
{passage}
---

**Generated Instruction:**
---
{instruction}
---

**Generated Question:**
---
{question}
---

**Reference (Ground Truth) Question Type:**
{ref_type}

**Intended Difficulty (from source):**
{intended_diff}

**EVALUATION CRITERIA:**

1.  **Answerability (Score 0-10):** How answerable is the *Generated Question* using *only* the provided *Passage*?
    * 0: Impossible to answer or requires external knowledge.
    * 10: The answer is clearly and explicitly in the passage.

2.  **Clarity (Score 0-10):** How clear, grammatically correct, and unambiguous 
    is the *Generated Question* (and its *Instruction*)?
    * 0: Incomprehensible or completely broken grammar.
    * 10: Perfectly clear, fluent, and well-formed.

3.  **Relevance (Score 0-10):** How relevant is the *Generated Question* to the 
    main ideas of the *Passage*?
    * 0: Completely irrelevant or focused on a trivial detail.
    * 10: Directly addresses a key concept or important detail from the passage.

4.  **Difficulty Assessment (Easy/Medium/Hard):** What is the *actual* difficulty 
    of answering this question based on the passage?
    Respond with ONLY one word: Easy, Medium, or Hard.

5.  **Question Type Classification (String):** What is the *actual* question type 
    of the *Generated Instruction* and *Generated Question*? Choose *only* from this list:
    {question_types_json}
    Pay attention more to *Generated Instruction* column to classify the question type.
    
    ### Quick Guide to the Six Types
    - **readingMatchingFeatures** — *“Match the statements/features to options (people/researchers/years/…).”* Answers often letters; some options may be used more than once.
    - **readingMultipleChoices** — *“Choose the correct letter A–D / Which of the following…”* Standard multiple-choice with one best answer.
    - **readingShortAnswer** — WH-questions requiring a short phrase/word/number. Often with a word limit (e.g., “NO MORE THAN X WORDS AND/OR A NUMBER”). 
    - **readingTextCompletion** — Fill-in-the-gap in **text/sentences**. If the original instruction mentions a more specific format (summary/table/flow-chart), still classify as **readingTextCompletion** here (since the allowed list is limited to six types).
    - **readingTrueFalseNotGiven** — Verify statements against **facts** in the passage: **True / False / Not Given**.
    - **readingYesNoNotGiven** — Verify statements against **writer’s views/claims**: **Yes / No / Not Given**.

    **Disambiguation tips:**
    - If the instruction explicitly says “Match … to …” ⇒ `readingMatchingFeatures`.
    - If the instruction says “Choose the correct letter / Which of the following …” ⇒ `readingMultipleChoices`.
    - If it is a direct WH-question expecting a short phrase/number ⇒ `readingShortAnswer`.
    - If it says “Complete the text/sentences/gaps …” ⇒ `readingTextCompletion`.
    - **TFNG vs YNNG**: facts/information ⇒ `readingTrueFalseNotGiven`; writer’s views/claims ⇒ `readingYesNoNotGiven`.

**OUTPUT (JSON Only):**
Return only the JSON object:
{{
  "answerability_score": <int>,
  "clarity_score": <int>,
  "relevance_score": <int>,
  "judged_difficulty": "<Easy/Medium/Hard>",
  "judged_question_type": "<Type from list>"
}}
"""

def pick_model():
    for m in PREFERRED_MODELS:
        if m:
            return m
    return "gpt-4.1-mini"

def clean_json_response(text: str):
    if not text:
        return None
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if match:
        return match.group(0)
    print(f"\nWarning: Could not find JSON block in response:\n{text[:300]}\n")
    return None

def call_model_with_retry(client, model, prompt, max_attempts=3, base_sleep=5):
    last_err = None
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format=RESPONSE_FORMAT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp
        except Exception as e:
            last_err = e
            msg = str(e).upper()
            if any(x in msg for x in ["RATE", "TEMPORARILY", "TIMEOUT", "OVERLOADED", "UNAVAILABLE", "RETRY"]):
                sleep_s = base_sleep * (2 ** attempt)
                print(f"Transient error: {e}. Retry in {sleep_s}s... (Attempt {attempt+1}/{max_attempts})")
                time.sleep(sleep_s)
                continue
            break
    raise last_err if last_err else RuntimeError("Unknown error calling OpenAI")

def get_first_choice_text(resp):
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""

def pick_value(row, candidates, required=True):
    """Pick first existing column name from a list of candidates."""
    for name in candidates:
        if name in row and pd.notna(row[name]):
            return str(row[name])
    if required:
        raise KeyError(f"Missing required columns: {candidates}")
    return ""

def evaluate_sample(row_tuple, client, model_name, prompt_template, q_types_json):
    index, row = row_tuple
    try:
        # Robust column picking (handles different CSV schemas)
        passage = pick_value(row, ["reference_passage", "passage", "context", "text"])[:1500]
        instruction = pick_value(row, ["generated_instruction", "instruction", "prompt"])
        question = pick_value(row, ["generated_question", "question"])
        ref_type = pick_value(row, ["reference_question_type", "ref_type", "question_type"])
        intended_diff = pick_value(row, ["reference_difficulty", "intended_diff", "difficulty"])
    except KeyError as e:
        return index, {"error": f"Missing column: {e}", "raw_response": ""}

    prompt = prompt_template.format(
        passage=passage,
        instruction=instruction,
        question=question,
        ref_type=ref_type,
        intended_diff=intended_diff,
        question_types_json=q_types_json,
    )

    try:
        resp = call_model_with_retry(client, model_name, prompt)
        raw = get_first_choice_text(resp)

        diag = {}
        try:
            diag["usage_prompt_tokens"] = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
            diag["usage_completion_tokens"] = getattr(getattr(resp, "usage", None), "completion_tokens", None)
            diag["usage_total_tokens"] = getattr(getattr(resp, "usage", None), "total_tokens", None)
        except Exception:
            pass

        json_str = clean_json_response(raw)
        if json_str:
            try:
                result_json = json.loads(json_str)
                # Validate keys
                for k in [
                    "answerability_score",
                    "clarity_score",
                    "relevance_score",
                    "judged_difficulty",
                    "judged_question_type",
                ]:
                    if k not in result_json:
                        raise KeyError(f"Missing key in JSON: {k}")
                result_json["error"] = None
                result_json.update(diag)
                return index, result_json
            except Exception as e:
                return index, {
                    "error": f"JSONParseError: {e}",
                    **diag,
                    "raw_response": raw[:800],
                }
        else:
            return index, {
                "error": "No JSON found in response",
                **diag,
                "raw_response": raw[:800],
            }

    except Exception as e:
        return index, {"error": str(e), "raw_response": "N/A"}

def main():
    if not OPENAI_API_KEY:
        print("=" * 60)
        print("ERROR: Missing OPENAI_API_KEY environment variable.")
        print("Set it and re-run.")
        print("=" * 60)
        return

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        model_name = pick_model()
        print(f"Using OpenAI model: {model_name}")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        if len(df) == 0:
            print(f"Loaded 0 rows from {INPUT_FILE}. Nothing to evaluate.")
            return
        print(f"Loaded {len(df)} samples from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Starting evaluation on {len(df)} samples...")

    all_results = [None] * len(df)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                evaluate_sample,
                row_tuple,
                client,
                model_name,
                PROMPT_TEMPLATE,
                QUESTION_TYPES_JSON,
            ): row_tuple[0]
            for row_tuple in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(df), desc="Evaluating Samples"):
            index, result_dict = future.result()
            all_results[index] = result_dict

    print("Evaluation complete. Processing and saving results...")

    df_results = pd.DataFrame(all_results)
    df_final = pd.concat([df.reset_index(drop=True), df_results], axis=1)

    try:
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully saved all {len(df_final)} results to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("--- EVALUATION SUMMARY ---")
    print("=" * 60)

    if "error" in df_final.columns:
        errors = df_final["error"].notna().sum()
        successes = len(df_final) - errors
        print(f"Total Samples: {len(df_final)}")
        print(f"Successful:    {successes}")
        print(f"Errors:        {errors}")
    else:
        print(f"Total Samples: {len(df_final)}")
        print(f"Successful:    {len(df_final)}")
        print(f"Errors:        0")

    for col, label in [
        ("answerability_score", "Average Answerability"),
        ("clarity_score", "Average Clarity"),
        ("relevance_score", "Average Relevance"),
    ]:
        if col in df_final.columns and pd.api.types.is_numeric_dtype(df_final[col]):
            print(f"{label}: {df_final[col].mean():.2f} / 10")
        else:
            print(f"{label}: N/A (no data)")

    if "judged_difficulty" in df_final.columns:
        try:
            dist = df_final["judged_difficulty"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}")
            print("\n--- Judged Difficulty Distribution ---")
            print(dist)
        except Exception:
            print("\n--- Judged Difficulty Distribution ---")
            print("N/A")

    if "judged_question_type" in df_final.columns:
        try:
            dist = df_final["judged_question_type"].value_counts(normalize=True).apply(lambda x: f"{x:.1%}")
            print("\n--- Judged Question Type Distribution ---")
            print(dist)
        except Exception:
            print("\n--- Judged Question Type Distribution ---")
            print("N/A")

    # Crosstab (support either column name)
    intended_col = None
    for cand in ["reference_difficulty", "intended_diff", "difficulty"]:
        if cand in df_final.columns:
            intended_col = cand
            break

    if intended_col and "judged_difficulty" in df_final.columns:
        try:
            print("\n" + "=" * 60)
            print("--- DIFFICULTY CONTROL ANALYSIS ---")
            print("=" * 60)
            crosstab = pd.crosstab(
                df_final[intended_col],
                df_final["judged_difficulty"],
                margins=True
            )
            print(f"Rows: Intended Difficulty ({intended_col}) | Columns: Judged (Actual) Difficulty")
            print(crosstab)
        except Exception:
            pass

if __name__ == "__main__":
    main()
