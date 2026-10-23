"""Self-harm specialist v2: teach sentinet/suicidality that a threat at someone else is not self-harm.

The shipped specialist read film lines such as "I'm going to find you, and I'm going to kill you" as the
writer's own suicidality (0.96), so C2 raised urgent self-harm alerts on films and threats. The training
and evaluation side lives here; the app keeps loading the model named in `packs/policy/decision.json`.

    data.py         replay + Civil Comments threats + C2-written counterfactual pairs -> data/self_harm_v2/
    external.py     OpenAI moderation, XSTest and HateCheck, as tests only
    train.py        5-seed fine-tune, chosen on val -> evaluation/results/self_harm_v2/model_level.json
    engine_eval.py  shipped vs fine-tuned inside the full engine -> engine_level.json
The Colab notebook `notebooks/self_harm_v2_colab.ipynb` runs all four in order.
"""
