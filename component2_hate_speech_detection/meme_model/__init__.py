"""Meme hate classifier: frozen CLIP features + a small trained fusion head (SO3, FR5).

The training and evaluation side. It reads the Hateful Memes dataset (Kiela et
al., 2020) from the git-ignored `data/` folder. The app runs the exported model
through `src/engine/meme_scorer.py`, and its score enters a live decision only
once it passes the calibration gate in `engine/fusion.py` (I-1).

    embed.py      one pass of the frozen encoder over every meme -> data/embeddings/
    heads.py      fusion heads (concat / align / cross), after Hate-CLIPper
    train.py      train, calibrate and evaluate -> evaluation/results/
    calibrate.py  the I-1 calibration gate -> evaluation/results/*_calibration.json
"""
