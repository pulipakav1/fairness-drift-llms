import os
import time

import requests

from src.config import ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # this repo only, not parent folders
except ImportError:
    pass

MAX_TOKENS = 500
SEED = 42
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


def query_claude(model, prompt, temperature):
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(model=model, max_tokens=MAX_TOKENS, temperature=temperature,
                                      messages=[{"role": "user", "content": prompt}])
    return response.content[0].text.strip()


def query_openai(model, prompt, temperature):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(model=model, max_tokens=MAX_TOKENS, temperature=temperature,
                                              seed=SEED, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()


def query_gemini(model, prompt, temperature):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    # Category filters off; prompt blocks still reported
    safety = {category: "BLOCK_NONE" for category in [
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]}
    response = genai.GenerativeModel(model).generate_content(
        prompt, safety_settings=safety,
        generation_config={"temperature": temperature, "max_output_tokens": MAX_TOKENS})
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return f"[BLOCKED: {response.prompt_feedback.block_reason}]"
    return response.text.strip() if response.text else "[No text returned]"


def query_hugging_face(model, prompt, temperature):
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": MAX_TOKENS, "temperature": temperature, "seed": SEED}
    response = requests.post(HF_CHAT_URL, json=payload, headers=headers, timeout=120)
    if response.status_code == 503:  # model loading; retry once
        time.sleep(response.json().get("estimated_time", 30))
        response = requests.post(HF_CHAT_URL, json=payload, headers=headers, timeout=120)
    if response.status_code != 200:
        return f"Error: HTTP {response.status_code}: {response.text[:200]}"
    return str(response.json()["choices"][0]["message"]["content"]).strip()


QUERY_FUNCTIONS = {
    "claude": query_claude,
    "openai": query_openai,
    "gemini": query_gemini,
    "llama31": query_hugging_face,
    "llama32": query_hugging_face,
    "gemma": query_hugging_face,
}


def query_model(provider, model, prompt, temperature=0.0):
    """Response text, or "Error: ..." if the call fails."""
    if provider not in QUERY_FUNCTIONS:
        return f"Error: unknown provider '{provider}'"
    try:
        return QUERY_FUNCTIONS[provider](model, prompt, temperature)
    except Exception as e:
        return f"Error: {e}"
