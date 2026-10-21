# Step 9 results (RQ3): resources

`evaluation/resources.py` on Intel64 Family 6 Model 186 Stepping 3, GenuineIntel (10 cores, 31.6 GB RAM), CPU only. Engine `lexicon+onnx-int8:roberta-hate-speech-dynabench-r4-target+onnx-int8:toxic-bert+onnx-int8:suicidality+onnx-int8:deberta-v3-xsmall-zeroshot-v1.1-all-33`.

| What | Result |
|---|---|
| Peak memory, every C2 model loaded | 1,847 MB (budget about 2,048 MB) |
| Typed message (288 texts) | 36.8 ms median, 161.5 ms p95 |
| ... on a grooming topic (112) | 138.3 ms median |
| Chat screenshot, whole frame read | 1,049 ms |
| Chat screenshot, one new message (partial read) | 243 ms |
| Chat screenshot, nothing changed | 11.0 ms (unchanged) |
| Meme image (30) | 321.6 ms median, 785.2 ms p95 |
| Speech model load (once, on first speech) | 555 ms |
| Spoken sentence, 2.8 s of speech | 382 ms |
| Spoken sentence, 2.9 s of speech | 377 ms |
| Spoken sentence, 2.6 s of speech | 347 ms |
| Internet connections held by the process | 0 (after every phase) |
| PyTorch imported | no |
