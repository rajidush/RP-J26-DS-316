"""Build packs/policy/words.json: the English words a text needs before a model's reading counts.

    python -m component2_hate_speech_detection.tools.build_word_list
    python -m component2_hate_speech_detection.tools.build_policy_manifest

The words are the whole-word entries of 3+ letters in the bert-base-uncased
WordPiece vocabulary (Apache-2.0), read from toxic-bert's exported tokenizer.
Sub-word pieces ("##ing") and single letters are left out. The list only has
to tell language from OCR garbage ("OLMPCCHW", "A-A∨±S-"), not be complete:
slang and slurs it lacks are the keyword layer's job.
"""
from __future__ import annotations

import json
import re

from component2_hate_speech_detection.src.engine.onnx_backend import onnx_model_dir
from component2_hate_speech_detection.src.engine.policy import PACK_DIR

SOURCE_MODEL = "unitary/toxic-bert"
MIN_LETTERS = 3


def main() -> None:
    vocab = json.loads((onnx_model_dir(SOURCE_MODEL) / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    words = sorted(w for w in vocab if re.fullmatch(rf"[a-z]{{{MIN_LETTERS},}}", w))
    pack = {
        "pack": "c2-words",
        "version": "1.0.0",
        "source": "bert-base-uncased WordPiece vocabulary (Apache-2.0), whole words of 3+ letters, via toxic-bert's tokenizer",
        "min_letters": MIN_LETTERS,
        "words": words,
    }
    # One word per line, so a change to the list is a readable diff.
    text = json.dumps(pack, indent=0, ensure_ascii=True)
    (PACK_DIR / "words.json").write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"{len(words)} words -> {PACK_DIR / 'words.json'}")


if __name__ == "__main__":
    main()
