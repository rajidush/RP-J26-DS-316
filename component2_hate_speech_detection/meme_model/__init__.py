"""Meme hate classifier: frozen CLIP features + a small trained fusion head (SO3, FR5).

Research build only. It reads the Hateful Memes dataset (Kiela et al., 2020)
from the git-ignored `data/` folder, and nothing it produces enters a live
decision until it passes the calibration gate in `engine/fusion.py` (I-1).

    embed.py   one pass of the frozen encoder over every meme -> data/embeddings/
    heads.py   fusion heads (concat / align / cross), after Hate-CLIPper
    train.py   train, calibrate and evaluate -> evaluation/results/
"""
