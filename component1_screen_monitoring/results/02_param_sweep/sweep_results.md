## Parameter sweep (selected on validation; test scored once)

| config | unfreeze | lr | epochs | trainable params | val acc | val recall | val F1 | min |
|---|---|---|---|---|---|---|---|---|
| baseline (no training) | - | - | 0 | 0 | 0.9958 | 0.9977 | 0.9960 | 0.0 |
| head_only | 0 | 0.0001 | 3 | 1,538 | 0.9934 | 0.9920 | 0.9937 | 3.5 |
| u2_lr1e-5 | 2 | 1e-05 | 3 | 14,178,818 | 0.9880 | 0.9817 | 0.9885 | 3.5 |
| u2_lr3e-5 | 2 | 3e-05 | 3 | 14,178,818 | 0.9922 | 0.9897 | 0.9926 | 3.4 |
| u2_lr5e-5 | 2 | 5e-05 | 3 | 14,178,818 | 0.9922 | 0.9897 | 0.9926 | 3.3 |
| u4_lr3e-5 | 4 | 3e-05 | 3 | 28,354,562 | 0.9934 | 0.9920 | 0.9937 | 3.5 |
| full_lr1e-5 | 12 | 1e-05 | 3 | 85,057,538 | 0.9934 | 0.9920 | 0.9937 | 4.8 |

Test set (n=1661):

| model | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| baseline (pretrained) | 0.9964 | 0.9989 | 0.9943 | 0.9966 |
| section 7 fine-tune (u2, lr 3e-5, 5 ep) | 0.9910 | 0.9988 | 0.9840 | 0.9914 |
| best of sweep: head_only | 0.9952 | 0.9989 | 0.9920 | 0.9954 |