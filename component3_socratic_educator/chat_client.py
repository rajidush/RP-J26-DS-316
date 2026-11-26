"""
chat_client.py

One way for the PP2 data scripts (simulate_children.py, draft_pp2_dataset.py) to
reach a chat model through an OpenAI-compatible endpoint:

    colab     Colab's model proxy (after `from google.colab import ai`)
    aistudio  AI Studio's free Gemini key (GEMINI_API_KEY)
    hf        Hugging Face Inference Providers (HF_TOKEN)
    local     a server on this machine, such as LM Studio (--base-url)

`make_chat` returns `ask(messages) -> text`; `ask.served` collects the models that
answered, for the run record. The `openai` package is imported only when a client
is made, so tests can replace it.
"""
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from draft_inquire_dataset import model_endpoint

BACKENDS = ("colab", "aistudio", "hf", "local")
LOCAL_BASE_URL = "http://127.0.0.1:1234/v1"   # LM Studio's OpenAI-compatible server
HF_BASE_URL = "https://router.huggingface.co/v1"   # Hugging Face Inference Providers, with HF_TOKEN


def base_url_of(backend, base_url=None):
    """The address a backend talks to, for the run record."""
    return {"local": base_url or LOCAL_BASE_URL, "hf": HF_BASE_URL}.get(backend, "fixed by the backend")


def make_chat(backend, model, temperature, max_tokens, base_url=None):
    from openai import OpenAI
    if backend == "local":
        base_url, key = base_url or LOCAL_BASE_URL, "local"
    elif backend == "hf":
        base_url, key = HF_BASE_URL, os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN not set: a Hugging Face "
                                                                             "access token with inference rights")
    else:
        base_url, key = model_endpoint(backend)
    client = OpenAI(base_url=base_url, api_key=key, timeout=120, max_retries=3)

    def ask(messages):
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                  max_tokens=max_tokens)
        ask.served.add(getattr(response, "model", None) or "not recorded")
        return response.choices[0].message.content or ""
    ask.served = set()
    return ask


# A bad request, a bad key or an unknown model will fail the same way however long one waits.
NOT_RESUMABLE = (400, 401, 404)


def is_resumable(error):
    """Whether waiting and rerunning can help: a quota (402, 403, 429) or an overload (5xx) can; a bad request,
    key or model name (400, 401, 404) cannot."""
    return getattr(error, "status_code", None) not in NOT_RESUMABLE


def runtime():
    """Python, platform and openai version for a run record ("not recorded" when unknown)."""
    try:
        openai_version = version("openai")
    except PackageNotFoundError:
        openai_version = "not recorded"
    return {"python": platform.python_version(), "platform": platform.platform(), "openai": openai_version}
