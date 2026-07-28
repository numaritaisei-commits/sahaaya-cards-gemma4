# Sahaaya Cards Gemma 4 runtime smoke

Private, Internet-disabled, single-P100 compatibility smoke for KerasHub 0.28.0 and the official Keras Gemma 4 E2B instruction preset.

The script verifies the pinned wheel SHA-256, safe ZIP inventory and every RECORD hash before atomic extraction. It then proves the vendored import origin/version, rejects non-P100 execution, checks the model tree without loading weight files locally, and performs one bounded float16 greedy generation. Generated text is never printed or saved; only its length and SHA-256 enter the report.

Inputs are limited to the competition, the future private verified-wheel Dataset, and `keras/gemma4/keras/gemma4_instruct_2b/2`. No network, pip, subprocess, dynamic import, eval, or exec path exists.

Local verification:

```text
python -m unittest discover -s tests -v
python validate_smoke.py gemma4_smoke.py kernel-metadata.example.json
```

After the Private Kaggle run, validate the downloaded report and atomic journal
together. A kernel `COMPLETE` state is not sufficient:

```text
python validate_smoke.py gemma4_smoke.py kernel-metadata.example.json \
  --report /path/to/gemma4_smoke_report.json \
  --journal /path/to/gemma4_smoke_journal.json
```

The output gate rejects duplicate JSON keys, unexpected fields or sources,
wheel/model inventory drift, non-P100 runtime, generated-body leakage, and a
journal digest that does not exactly match the report.

This directory has not been pushed or executed on Kaggle.
