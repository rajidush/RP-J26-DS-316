
# model_client.py -- standalone test that the local SLM responds from code.
# Used by grammar_decoder.py for INTERCEPT; `response_format` carries the
# JSON schema that LM Studio enforces while generating (ADR 0004).

import time
import urllib.request
import json

from . import config

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

def call_local_model(prompt: str, timeout: float = 30.0, response_format: dict | None = None,
                     max_tokens: int = 60) -> tuple[str, float]:
    payload = {
        "model": config.LOCAL_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    # Without a timeout a stalled server hangs the whole FSM session; on timeout
    # the caller (grammar_decoder.generate) falls back to the template pool.
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - start
    return data["choices"][0]["message"]["content"], elapsed

if __name__ == "__main__":
    text, seconds = call_local_model("Say hello in one short sentence.")
    print(f"Response ({seconds:.2f}s): {text}")