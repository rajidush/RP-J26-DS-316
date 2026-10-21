## Screen fine-tuning sweep (start: `jaranohaal/vit-base-violence-detection` pretrained; seed 42; 3 epochs, fp16, cosine; RLVS share 40% of 10302 samples/epoch; weighted CE [0.83, 1.259])

Screen val n=1078 (224 composites + 854 tiles, violent 381); RLVS val n=1661. Threshold 0.5. Chosen = best screen-val F1 with RLVS-val accuracy ≥ 98%. The screen TEST set was not used.

| run | unfreeze | lr | screen P | screen R | screen F1 | composites F1 | tiles F1 | missed | false alarms | RLVS val acc | RLVS val F1 | ≥ floor | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (pretrained, no training) | - | - | 0.910 | 0.160 | 0.272 | 0.035 | 0.353 | 320 | 6 | 0.9958 | 0.996 | yes | 0.0 |
| u2_lr1e-5 | 2 | 1e-05 | 0.906 | 0.381 | 0.536 | 0.271 | 0.623 | 236 | 15 | 0.9904 | 0.991 | yes | 5.4 |
| u2_lr3e-5 | 2 | 3e-05 | 0.857 | 0.614 | 0.716 | 0.600 | 0.756 | 147 | 39 | 0.9910 | 0.991 | yes | 5.3 |
| u4_lr3e-5 ★ | 4 | 3e-05 | 0.895 | 0.693 | 0.781 | 0.659 | 0.824 | 117 | 31 | 0.9922 | 0.993 | yes | 5.5 |

### Threshold sweep — u4_lr3e-5 on screen val

| rule | threshold | screen P | screen R | screen F1 | missed | false alarms | RLVS val acc |
|---|---|---|---|---|---|---|---|
| recall>=0.9 | 0.05 | 0.630 | 0.913 | 0.746 | 33 | 204 | 0.9928 |
| max F1 | 0.25 | 0.841 | 0.793 | 0.816 | 79 | 57 | 0.9916 |
| default | 0.50 | 0.895 | 0.693 | 0.781 | 117 | 31 | 0.9922 |
