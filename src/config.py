from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "outputs" / "results" / "raw_results_20251115_035107.csv"
SCORES_PATH = ROOT / "outputs" / "results" / "scored_responses.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "figures"

DATASETS = ["bbq", "bold", "crows_pairs", "holistic_bias", "realtoxicityprompts", "stereoset", "winobias"]
PROMPTS_PER_DATASET = 100
METRICS = ["sentiment_score", "toxicity_score", "stereotype_score"]

SEED = 42
N_BOOT = 1000
ALPHA = 0.05
MIN_GROUP_SIZE = 5  # min pairs per demographic test

# Dropped in the ablation
ABLATION_EXCLUDED = ["stereoset", "crows_pairs"]

# Oldest first; this order fixes the bootstrap draws
FAMILIES = {
    "openai": ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "claude_sonnet": ["claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929"],
    "claude_opus": ["claude-opus-4-1-20250805"],
    "gemini": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    "llama31": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
    ],
    "llama32": ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"],
    "gemma": ["google/gemma-2-2b-it", "google/gemma-2-9b-it"],
}

# Provider key in the results files
PROVIDERS = {
    "openai": "openai", "claude_sonnet": "claude", "claude_opus": "claude", "gemini": "gemini",
    "llama31": "llama31", "llama32": "llama32", "gemma": "gemma",
}

# Release chains: 11 models, 8 transitions
RELEASE_CHAINS = {family: FAMILIES[family] for family in ("openai", "claude_sonnet", "gemini")}

# Same-tier transitions
STRICT_TRANSITIONS = {
    ("openai", "gpt-4-turbo", "gpt-4o"),
    ("claude_sonnet", "claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929"),
    ("gemini", "gemini-2.0-flash", "gemini-2.5-flash"),
}

# Descriptive only, no BH
CONTEXT_TRANSITIONS = [
    ("claude_cross_tier", "claude-sonnet-4-20250514", "claude-opus-4-1-20250805"),
    ("claude_cross_tier", "claude-opus-4-1-20250805", "claude-sonnet-4-5-20250929"),
    ("llama31_scale", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"),
    ("llama31_scale", "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-405B-Instruct"),
    ("llama32_scale", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("gemma_scale", "google/gemma-2-2b-it", "google/gemma-2-9b-it"),
]

# Unlisted labels are excluded
AXIS_MAP = {
    "gender": "Gender", "gender_identity": "Gender",
    "race": "Race/Ethnicity", "race-color": "Race/Ethnicity",
    "religion": "Religion",
    "age": "Age", "Age": "Age",
    "disability": "Disability", "ability": "Disability",
    "nationality": "Nationality",
    "socioeconomic": "Socioeconomic",
}

PROVIDER_GROUPS = {
    "OpenAI": ["openai"],
    "Anthropic": ["claude"],
    "Google (Gemini)": ["gemini"],
    "Meta (LLaMA)": ["llama31", "llama32"],
    "Google (Gemma)": ["gemma"],
}

# Plot labels
METRIC_LABELS = {"sentiment_score": "Sentiment", "toxicity_score": "Toxicity", "stereotype_score": "Stereotype"}
DATASET_LABELS = {
    "bbq": "BBQ", "bold": "BOLD", "crows_pairs": "CrowS-Pairs", "holistic_bias": "HolisticBias",
    "realtoxicityprompts": "RealToxicityPrompts", "stereoset": "StereoSet", "winobias": "WinoBias",
}
MODEL_LABELS = {
    "gpt-4-turbo": "GPT-4-turbo", "gpt-4o": "4o", "gpt-4o-mini": "4o-mini", "gpt-4.1": "4.1",
    "gpt-4.1-mini": "4.1-mini",
    "claude-sonnet-4-20250514": "Sonnet 4", "claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "claude-opus-4-1-20250805": "Opus 4.1",
    "gemini-2.0-flash": "Gemini 2.0-Flash", "gemini-2.5-flash": "2.5-Flash",
    "gemini-2.5-flash-lite": "2.5-Flash-Lite", "gemini-2.5-pro": "2.5-Pro",
    "meta-llama/Llama-3.1-8B-Instruct": "LLaMA-3.1 8B", "meta-llama/Llama-3.1-70B-Instruct": "70B",
    "meta-llama/Llama-3.1-405B-Instruct": "405B",
    "meta-llama/Llama-3.2-1B-Instruct": "LLaMA-3.2 1B", "meta-llama/Llama-3.2-3B-Instruct": "3B",
    "google/gemma-2-2b-it": "Gemma-2 2B", "google/gemma-2-9b-it": "9B",
}


def release_transitions():
    """Yield (family, old_model, new_model) for the 8 release transitions."""
    for family, chain in RELEASE_CHAINS.items():
        for old_model, new_model in zip(chain, chain[1:]):
            yield family, old_model, new_model
