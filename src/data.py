import json

import pandas as pd

from src.config import PROMPTS_PER_DATASET, RAW_PATH, ROOT, SCORES_PATH, TABLES_DIR

CROWS_PAIRS_PATH = ROOT / "data" / "crows_pairs_anonymized.csv"


def load_raw(path=RAW_PATH):
    return pd.read_csv(path, low_memory=False)


def load_scores(path=SCORES_PATH):
    return pd.read_csv(path, low_memory=False)


def save_table(df, name):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / name, index=False, lineterminator="\n")  # same bytes on every OS
    print(f"Saved {name}")


DROPPED = {"empty", "api_error", "policy_blocked", "no_text"}

REFUSAL_PATTERNS = [
    "i cannot",
    "i'm sorry, i'm unable",
    "i'm not able to",
    "i am unable to",
    "as an ai, i cannot",
    "as an ai i cannot",
    "i apologize, but i",
    "i'm afraid i can't",
]


def response_issue(response):
    """Issue label for a response, or None if it is clean."""
    if not isinstance(response, str) or response == "":
        return "empty"
    if response.startswith("Error:"):
        return "api_error"
    if response.startswith("[BLOCKED:"):
        return "policy_blocked"
    if response == "[No text returned]":
        return "no_text"
    if any(pattern in response.lower() for pattern in REFUSAL_PATTERNS):
        return "refusal"
    return None


def filter_responses(raw):
    """Drop unscorable rows; keep refusals."""
    issues = raw["response"].map(response_issue)
    return raw[~issues.isin(DROPPED)].reset_index(drop=True)


# Benchmark prompts (collection only); list position = prompt_id

def _record(prompt, **metadata):
    return {"prompt": prompt, **{k: v for k, v in metadata.items() if v is not None}}


def _hf_rows(name, n, **kwargs):
    from datasets import load_dataset
    return load_dataset(name, **kwargs).select(range(n))


def load_bold(n=PROMPTS_PER_DATASET):
    try:
        rows = _hf_rows("AlexaAI/bold", n, split="train")
    except ValueError as e:
        # Fallback when `datasets` can't parse BOLD
        if "Feature type 'List' not found" not in str(e):
            raise
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo_id="AlexaAI/bold", filename="train.json"), encoding="utf-8") as f:
            rows = json.load(f)[:n]
    records = []
    for row in rows:
        prompts = row.get("prompts")
        if not prompts:
            continue
        records.append(_record(prompts[0] if isinstance(prompts, list) else prompts,
                               axis=row.get("category") or "bold",
                               group=row.get("target_group") or row.get("subcategory") or row.get("target"),
                               domain=row.get("domain") or row.get("category")))
    return records


def load_stereoset(n=PROMPTS_PER_DATASET):
    rows = _hf_rows("McGill-NLP/stereoset", n, name="intersentence", split="validation")
    return [_record(row["context"], axis=row.get("bias_type") or "stereoset", group=row.get("target"),
                    bias_type=row.get("bias_type"), target=row.get("target"))
            for row in rows if row.get("context")]


def load_bbq(n=PROMPTS_PER_DATASET):
    records = []
    for row in _hf_rows("heegyu/bbq", n, name="Age", split="test"):
        text = (row.get("context", "") + " " + row.get("question", "")).strip()
        if not text:
            continue
        axis = row.get("category") or "bbq"
        group = row.get("protected_value") or row.get("protected_group") or row.get("attribute") or axis
        records.append(_record(text, axis=axis, group=group, category=row.get("category"),
                               question=row.get("question"), answer_info=row.get("answer_info")))
    return records


def load_crows_pairs(n=PROMPTS_PER_DATASET):
    if not CROWS_PAIRS_PATH.exists():
        raise FileNotFoundError(f"{CROWS_PAIRS_PATH} not found. Download crows_pairs_anonymized.csv from "
                                "https://github.com/nyu-mll/crows-pairs into data/.")
    rows = pd.read_csv(CROWS_PAIRS_PATH).head(n)
    return [_record(row["sent_more"], axis=row.get("bias_type") or "crows_pairs", group=row.get("category"),
                    bias_type=row.get("bias_type"), sent_less=row.get("sent_less"))
            for _, row in rows.iterrows() if isinstance(row["sent_more"], str) and row["sent_more"].strip()]


def load_real_toxicity_prompts(n=PROMPTS_PER_DATASET):
    rows = _hf_rows("allenai/real-toxicity-prompts", n, split="train")
    return [_record(row["prompt"]["text"], axis=row["prompt"].get("category") or "toxicity",
                    group=row["prompt"].get("subcategory") or row.get("challenger"),
                    prompt_toxicity=row["prompt"].get("toxicity"))
            for row in rows if row["prompt"].get("text")]


def load_holistic_bias(n=PROMPTS_PER_DATASET):
    rows = _hf_rows("fairnlp/holistic-bias", n, data_files=["sentences.csv"], split="train")
    return [_record(row.get("sentence") or row.get("text"),
                    axis=row.get("axis") or row.get("category") or "holistic_bias",
                    group=row.get("descriptor") or row.get("group") or row.get("target"),
                    axis_source=row.get("axis"))
            for row in rows if row.get("sentence") or row.get("text")]


def load_winobias(n=PROMPTS_PER_DATASET):
    rows = _hf_rows("Elfsong/Wino_Bias", n, split="train")
    return [_record(row["input"], axis="gender_identity", group=row.get("gender"), polarity=row.get("polarity"),
                    reference=row.get("reference"), wb_type=row.get("type"))
            for row in rows if row.get("input")]


LOADERS = {
    "bold": load_bold,
    "stereoset": load_stereoset,
    "bbq": load_bbq,
    "crows_pairs": load_crows_pairs,
    "realtoxicityprompts": load_real_toxicity_prompts,
    "holistic_bias": load_holistic_bias,
    "winobias": load_winobias,
}
