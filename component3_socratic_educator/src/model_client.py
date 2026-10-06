
# model_client.py -- standalone test that the local SLM responds from code.
# Not wired into grammar_decoder.py yet -- that's Week 3.

import time
import urllib.request
import json

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

def call_local_model(prompt: str, timeout: float = 30.0) -> tuple[str, float]:
    payload = {
        "model": "gemma-3-1b",  # match the model name shown in LM Studio
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
    }
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