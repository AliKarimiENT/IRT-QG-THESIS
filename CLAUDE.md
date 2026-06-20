# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a master's thesis research project: **IRT-controlled Neural Question Generation (NQG) for IELTS reading**. The central idea is to steer a seq2seq language model to generate IELTS reading questions of a *controllable difficulty* by injecting Item Response Theory (IRT) parameters into the model's input as control signals. The proposed model is a LoRA-fine-tuned **Flan-T5**; **T5-base**, **BART-base**, and **GPT-2** are trained as baselines for comparison. Quality is measured with an LLM-as-judge protocol.

This is a research codebase, not an application. The "source of truth" for the actual experiments is the **Jupyter notebooks** (run on Kaggle/Colab with GPU), not the small `.py` files — see "Two layers of code" below.

## The IRT control mechanism (the core contribution)

Categorical difficulty (`Easy`/`Medium`/`Hard`) is mapped to the IRT difficulty parameter `b` via a fixed proxy table (`Easy→-1.0`, `Medium→0.0`, `Hard→1.0`), with discrimination `a=1.0` and guessing `c=0.0` held constant. The model input ("source") is formatted as:

```
[a=1.0, b=0.0, c=0.0] <task prefix> <TYPE_READINGSHORTANSWER>
Passage: {passage text}
```

The target output deliberately contains **no** control tokens — only the content the model should learn to produce:

```
Instruction: ...
Question: ...
Answer: ...
```

Putting controls only on the input lets the model focus on content while difficulty/type are conditioned externally. At inference, you set the difficulty + question type to steer generation. The full design is documented step-by-step in `src/pseudocode/` (`0. MAIN.md` … `10. INFERENCE.md`) — **read these first** to understand the pipeline; they mirror what the FlanT5-IRT notebook implements.

## Key domain facts

- **Six IELTS reading question types** are the controlled vocabulary throughout: `readingMatchingFeatures`, `readingMultipleChoices`, `readingShortAnswer`, `readingTextCompletion`, `readingTrueFalseNotGiven`, `readingYesNoNotGiven`. These appear as `<TYPE_*>` special tokens, as task prefixes, and as the judge's classification labels.
- **Dataset:** `src/data/final_data.json` — 2601 IELTS reading records. Fields: `Topic`, `PassageContext`, `QuestionType`, `InstructionContext`, `Question`, `Answer`, `category`, `difficulty`, `category_confidence`, `difficulty_confidence`. These map to canonical names (`passage`, `question_type`, `ref_instruction`, `ref_question`, `ref_answer`) during preparation.
- **No passage leakage:** train/val/test splits are *grouped by passage* (`passage_id` = MD5 of passage text) so all questions from one passage stay in the same split. Never write a split that shuffles at the row level.
- **Special tokens** added to the tokenizer: `<SEP>`, `<BLANK>`, plus one `<TYPE_*>` per question type; the model's embeddings are resized after adding them.

## Architecture: training & evaluation flow

1. **Prepare** — load `final_data.json`, normalize schema, derive `passage_id`, normalize special tokens (`BLANK`→`<BLANK>`), filter (passage ≥ 50 chars, non-empty instruction/question).
2. **Format** — build IRT-controlled source + structured target (above); wrap in a `QGDataset` (`torch.utils.data.Dataset`) that tokenizes with dynamic padding via `DataCollatorForSeq2Seq`.
3. **Train** — `Seq2SeqTrainer` with LoRA (`r=16, alpha=32, dropout=0.05`, target modules `q,k,v,o,wi_0,wi_1,wo`), fp16, label smoothing, early stopping on `eval_loss`, `predict_with_generate=True`. Beam search (`num_beams=4`).
4. **Generate** — each fine-tuning notebook writes `generated_samples*.csv` (per-model variants exist: `_T5`/`_BART`/`_GPT2`, with the unsuffixed file = the proposed Flan-T5 model).
5. **Judge** — `src/evaluation/judge_eval.py` runs an OpenAI model as judge over the generated CSV, scoring **answerability / clarity / relevance** (0–10), plus **judged difficulty** and **judged question type**. Output: `llm_judged_results*.csv`.
6. **Analyze** — `src/evaluation/NQG-evaluation.ipynb` aggregates the judged results into the charts in `src/evaluation/charts/`. The headline analysis is a **crosstab of intended vs. judged difficulty** — i.e. how well the IRT control actually worked.

## Two layers of code (important)

- **`.py` files are minimal scaffolding, not the real pipeline.** `src/config.py`, `src/data/prepare.py`, and `src/utils/io_utils.py` only build/handle a *toy 2-row sample* with a *different schema* (`context/answer/question/difficulty`) for local smoke-testing. Do not mistake `prepare.py` for the real data-prep logic — that lives in the notebooks and is specified in `src/pseudocode/`.
- **The notebooks are the experiments.** `src/finetuning/NQG-FlanT5-IRT.ipynb` is the proposed model; `NQG-t5-base.ipynb`, `NQG-BART-base.ipynb`, `NQG-GPT2.ipynb` are baselines. They install their own deps inline (`peft`, `sentencepiece`, `accelerate`, `rouge-score`) and fetch data via `gdown`, and assume Kaggle paths like `/kaggle/working/`.

When changing the pipeline, keep the pseudocode docs, the FlanT5-IRT notebook, and any `.py` mirror in sync — they describe one design.

## Commands

There is no build, lint, or test suite. Setup and the runnable scripts:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # note: notebooks also need peft, sentencepiece, accelerate, openai, rouge-score (not in requirements.txt)

python -m src.data.prepare               # writes the toy data/processed/sample.csv (smoke test only)

# LLM-as-judge evaluation:
export OPENAI_API_KEY=...                 # required, or the script exits
python src/evaluation/judge_eval.py
```

`judge_eval.py` has **hardcoded paths** (`INPUT_FILE = /kaggle/working/generated_samples.csv`, `OUTPUT_FILE = llm_judged_results.csv`) and a model preference list (`OPENAI_MODEL` env var → `gpt-4.1-mini` → `gpt-4o-mini` → …). Override `INPUT_FILE`/`OUTPUT_FILE` or the env var when running locally against files in `src/results/`. It uses `ThreadPoolExecutor` (3 workers) with exponential-backoff retry and robust column-name fallbacks, so it tolerates the differing CSV schemas across model variants.

The heavy training/generation work runs in the notebooks on GPU (Kaggle/Colab), not via these scripts.

## Models on Hugging Face

Trained model artifacts are published under the Hugging Face account **`alikarimiaca`** (https://huggingface.co/alikarimiaca). When loading a fine-tuned checkpoint rather than retraining, pull from there. Do **not** commit Hugging Face or OpenAI credentials/tokens to the repo.
