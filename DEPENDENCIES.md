# Dependency provenance

The repository contains no model weights and no wheel binary. The runtime is designed for a Private, Internet-disabled Kaggle notebook with only the sources listed below.

| Component | Pinned identity | Role | License / terms | Redistribution | Integrity |
|---|---|---|---|---|---|
| Gemma 4 Instruct E2B | Kaggle Models `keras/gemma4/keras/gemma4_instruct_2b/2` | Primary generation and verification intelligence | Gemma upstream terms | Not included | The notebook verifies the expected bounded file inventory before model load |
| KerasHub | 0.28.0, official PyPI pure-Python wheel | `Gemma4CausalLM` runtime | Apache-2.0 | Wheel not included | SHA-256 `a28cb601f7fffb7f28add1bae8110459fc3ac7d9e2159453dfbda9e97271fc87`; 766 ZIP members and 765 RECORD digests audited |
| Keras and Torch | Kaggle free GPU image | Keras/Torch two-pass app backend | Upstream terms | Not included | V4-V6 remained incomplete; no successful app runtime is claimed |
| Keras and JAX | JAX 0.7.2 and Keras 3.13.2 in the official Kaggle T4 x2 image | Separate model-parallel loading diagnostic, not the app | Upstream terms | Not included | Version 7 exact source, Private Internet-off metadata, official inputs, and extracted vendor passed validation; runtime ended `DIAGNOSTIC_FAILURE` / `ValueError` at `model_load_started` after 452.703s, before weight-load or generation success |
| Project validators and renderer | Python standard library | Static gate, evidence gate, offline HTML | Python license | Not included | No network, subprocess, dynamic import, package installation, or downloaded native executable path |

Primary format references:

- Google AI, Gemma 4 prompt formatting: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
- KerasHub Gemma 4 workflow guide: https://keras.io/keras_hub/guides/gemma4_multimodal_and_agentic_workflows/
- KerasHub `Gemma4CausalLM` API: https://keras.io/keras_hub/api/models/gemma4/gemma4_causal_lm/

The pinned wheel audit report, audit tool, and upstream license are included under `dependencies/` and `tools/`. The audit accepts only regular non-executable `.py` and `.txt` files plus the three named extensionless wheel metadata files; it rejects malformed directories, special files, native payloads, and unrecorded content. To reproduce a Kaggle run, obtain the exact wheel from its official source, verify it with `tools/audit_wheel.py`, place it in a Private Kaggle Dataset, and replace the owner placeholder consistently in the private notebook and metadata copies. The validator accepts the resulting safe owner without embedding any private username in this repository. Version 7 validates the separate JAX route's provenance, not model loading or application behavior. Do not commit the wheel, a filled owner, runtime diagnostics, or credentials.
