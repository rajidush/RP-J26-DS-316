"""
Finite-automaton token masking, shown step by step (proposal panel question:
"How will you implement finite automaton token masking?").

Loads the inference baseline's own weights (the QAT Q4_0 GGUF that LM Studio
serves to INTERCEPT) into transformers on the CPU, then writes the INTERCEPT
line twice with greedy decoding on the same prompt:

    unmasked  the model alone
    masked    with src/token_mask.py's automaton removing, before every token,
              each vocabulary entry that cannot continue a line the INTERCEPT
              controller grammar accepts

For every masked step it records the model's top choices before masking, which
of them the automaton blocked, and the probability mass removed. The
child-facing path is unchanged: it still uses LM Studio's JSON-schema
constraint (ADR 0004). This shows the same mechanism in code written here.

The prompt asks for a plain line, not JSON (the automaton enforces the line
itself), and, as in production, does not spell out the two openers.

Tokenizer: the one transformers rebuilds from the GGUF splits text badly
(" What were you" -> "▁W hat ▁w ere ▁y ou", 18 Oct check), so the prompt is
encoded with the Gemma 3 tokenizer from the local Hugging Face cache instead.
The run stops unless its vocabulary equals the GGUF's, id for id.

Needs `pip install gguf` (transformers reads GGUF through it). Run from
component3_socratic_educator/:
    python -m evidence.trace_token_masking [--model PATH]

Writes, in evidence/token_masking_<date>/, a .trace.json (every step), a
.summary.md (readable table), a .run_record.json and a .log.txt; never
overwrites a recorded run.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from src.grammar_decoder import INTERCEPT_PROMPT, delivery_violation
from src.token_mask import MaskingLogitsProcessor, TokenMask, piece_text

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = (Path.home() / ".lmstudio" / "models" / "lmstudio-community" / "gemma-3-1B-it-QAT-GGUF"
                 / "gemma-3-1B-it-QAT-Q4_0.gguf")
# Any cached repo with the Gemma 3 tokenizer will do; this one is already on the machine.
DEFAULT_TOKENIZER = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
MAX_NEW_TOKENS = 60
TOP_K = 5
# INTERCEPT_PROMPT up to its JSON instructions, then the same question
# instruction for a plain line.
TRACE_PROMPT = (INTERCEPT_PROMPT.split(" Return JSON.")[0]
                + " Write one short opening sentence, then one short, gentle, open question (under 15 words) "
                  "inviting the child to share what they were doing or thinking.")
GEMMA_TURN = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shown(text) -> str:
    return "(special)" if text is None else json.dumps(text, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="parent folder of the run folder (a trial run elsewhere)")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d")
    run_name = f"token_masking_{stamp}"
    run_dir = args.out / run_name
    outputs = {k: run_dir / f"{run_name}.{k}" for k in ("trace.json", "summary.md", "run_record.json", "log.txt")}
    if any(p.exists() for p in outputs.values()):
        sys.exit(f"{run_name} is already recorded; nothing was overwritten.")

    import gguf
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    field = gguf.GGUFReader(args.model).fields["tokenizer.ggml.tokens"]
    gguf_pieces = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
    pieces = tok.convert_ids_to_tokens(list(range(len(gguf_pieces))))
    # Compared as text: the GGUF writes the space-run tokens (ids 138-167) with real
    # spaces, the tokenizer with "▁".
    vocab_mismatches = sum(piece_text(a) != piece_text(b) for a, b in zip(pieces, gguf_pieces))
    if vocab_mismatches:
        sys.exit(f"{vocab_mismatches} token ids differ between {args.tokenizer} and the GGUF; nothing was recorded.")

    model = AutoModelForCausalLM.from_pretrained(args.model.parent, gguf_file=args.model.name,
                                                 torch_dtype=torch.float32)
    model.eval()
    load_seconds = time.time() - t0

    # One entry per row of the model's output scores; ids past the GGUF vocabulary have no text.
    n_scores = model.get_output_embeddings().weight.shape[0]
    special = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}
    texts = [piece_text(pieces[i], special=i in special) if i < len(pieces) else None for i in range(n_scores)]
    end_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")} - {None, tok.unk_token_id}
    mask = TokenMask(texts, end_ids)

    prompt_text = GEMMA_TURN.format(prompt=TRACE_PROMPT)
    input_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_len = input_ids.shape[1]
    settings = dict(do_sample=False, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=sorted(end_ids),
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)

    def generate(processors=None):
        t = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, logits_processor=processors, **settings)
        new = out[0, prompt_len:].tolist()
        while new and new[-1] in end_ids:
            new.pop()
        return new, time.time() - t

    unmasked_ids, unmasked_s = generate()
    processor = MaskingLogitsProcessor(mask, prompt_len=prompt_len, top_k=TOP_K)
    masked_ids, masked_s = generate(LogitsProcessorList([processor]))

    unmasked = tok.decode(unmasked_ids, skip_special_tokens=True).strip()
    masked = "".join(texts[i] or "" for i in masked_ids)
    for step in processor.trace:
        i = step["step"]
        step["chosen"] = ({"token_id": masked_ids[i], "text": texts[masked_ids[i]]} if i < len(masked_ids)
                          else {"token_id": None, "text": "(end of generation)"})
        for c in step["top_before_mask"]:
            c["piece"] = pieces[c["token_id"]]

    lines = {
        "unmasked": {"text": unmasked, "delivery_violation": delivery_violation(unmasked), "seconds": round(unmasked_s, 2)},
        "masked": {"text": masked, "delivery_violation": delivery_violation(masked), "seconds": round(masked_s, 2)},
    }
    blocked_top1 = [s for s in processor.trace if not s["top_before_mask"][0]["allowed"]]

    log = io.StringIO()
    with redirect_stdout(log):
        print(f"Model: {args.model}")
        print(f"Unmasked: {unmasked!r}  -> controller grammar/post-check: {lines['unmasked']['delivery_violation'] or 'pass'}")
        print(f"Masked:   {masked!r}  -> controller grammar/post-check: {lines['masked']['delivery_violation'] or 'pass'}")
        print(f"Steps where the model's first choice was blocked: {len(blocked_top1)} of {len(processor.trace)}\n")
        for s in processor.trace:
            top = s["top_before_mask"][0]
            mark = "BLOCKED" if not top["allowed"] else "kept   "
            print(f"step {s['step']:>2}  {mark} model's first choice {shown(top['text']):<22} p={top['prob']:.3f}"
                  f"  -> chose {shown(s['chosen']['text']):<22} allowed {s['allowed_count']:>6} tokens,"
                  f" removed p={s['blocked_probability']:.3f}")
    print(log.getvalue())

    summary = [f"# Finite-automaton token masking, {stamp}", "",
               f"Model: `{args.model.name}` (the INTERCEPT inference baseline's weights, loaded in transformers on CPU). "
               "Greedy decoding, same prompt for both runs. The child-facing path is unchanged (ADR 0004).", "",
               "| Run | Line | Controller grammar + post-check |", "|---|---|---|",
               *[f"| {k} | {v['text']!r} | {v['delivery_violation'] or 'pass'} |" for k, v in lines.items()], "",
               f"The model's first choice was blocked at {len(blocked_top1)} of {len(processor.trace)} steps.", "",
               "| Step | Text so far | Model's top choices before masking (✗ = blocked) | Chosen | Allowed tokens | Blocked p |",
               "|---|---|---|---|---|---|"]
    for s in processor.trace:
        tops = ", ".join(f"{'✗ ' if not c['allowed'] else ''}{shown(c['text'])} {c['prob']:.2f}" for c in s["top_before_mask"])
        summary.append(f"| {s['step']} | {shown(s['text_so_far'])} | {tops} | {shown(s['chosen']['text'])} | "
                       f"{s['allowed_count']} | {s['blocked_probability']:.3f} |")

    record = {
        "run": run_name, "started": started, "purpose": "show finite-automaton token masking on the INTERCEPT grammar",
        "model_file": str(args.model), "model_bytes": args.model.stat().st_size, "model_sha256": sha256(args.model),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "gguf": importlib.metadata.version("gguf"),
                    "accelerate": importlib.metadata.version("accelerate"),
                    "device": "cpu", "machine": platform.platform(), "processor": platform.processor() or "not recorded"},
        "prompt": TRACE_PROMPT, "chat_format": GEMMA_TURN.strip(), "prompt_tokens": prompt_len,
        "generation": {k: v for k, v in settings.items() if k != "pad_token_id"}, "top_k_recorded": TOP_K,
        "tokenizer": args.tokenizer, "gguf_vocabulary_size": len(gguf_pieces),
        "tokenizer_vs_gguf_vocabulary_mismatches": vocab_mismatches,
        "model_score_rows": n_scores, "end_token_ids": sorted(end_ids),
        "load_seconds": round(load_seconds, 2), "lines": lines,
        "steps": len(processor.trace), "steps_first_choice_blocked": len(blocked_top1),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs["trace.json"].write_text(json.dumps(processor.trace, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["summary.md"].write_text("\n".join(summary) + "\n", encoding="utf-8")
    outputs["run_record.json"].write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["log.txt"].write_text(log.getvalue(), encoding="utf-8")
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
