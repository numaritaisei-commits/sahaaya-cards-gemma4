# Security and privacy boundary

Sahaaya Cards is a fixed research prototype over two fictional notices. It is not a general upload service and does not authenticate, browse for, or contact an authority.

## Runtime boundary

- The Keras/Torch two-pass app uses a Private Kaggle notebook, free T4 x2, and
  Internet disabled. Its V4-V6 attempts remained incomplete, so this repository
  claims no successful app run.
- The separate JAX Version 7 smoke also used an exact Private,
  Internet-disabled T4 x2 run with only the official competition input, pinned
  official model, and audited wheel Dataset. Source, server metadata, and the
  extracted vendor tree passed validation; runtime ended after 452.703 seconds
  as `DIAGNOSTIC_FAILURE` / `ValueError` at `model_load_started`. Weight loading,
  generation, and application execution did not complete.
- Official Gemma 4 Kaggle Model plus one locally audited pure-Python KerasHub wheel Dataset.
- No API key, Cookie, email address, phone number, user document, or other personal data.
- No package installer, network fallback, subprocess, shell command, downloaded native executable, `eval`, or `exec`.
- Prompt-delimited notices and candidate responses are untrusted data, never instructions.

## Publication boundary

This repository intentionally excludes credentials, private Kaggle metadata,
wheel binaries, model weights, generated prompts and responses, runtime
journals, raw result artifacts, model/runtime videos, and local filesystem paths. This
includes the private Version 7 diagnostic files; only their bounded, validated
outcome is summarized here. `kernel-metadata.example.json` and
`dataset-metadata.example.json` contain owner placeholders only.
`ILLUSTRATIVE_OUTPUT.md` is a hand-authored interface fixture; it is explicitly
not a model output, runtime artifact, validator PASS, or translation-quality
claim.

The sole published video is the fixed-hash, 114-second silent fixture
walkthrough at `assets/sahaaya-cards-fixture-prototype.mp4`. It contains only
the two fictional notices and persistent evidence-boundary labels. It is not a
model-generated artifact or runtime-success claim. The validator enforces its
regular-file status, 20 MB bound, MP4 `ftyp` header, and exact SHA-256.

The public validator uses an exact source allowlist, rejects symlinks, verifies the integrity manifest, checks the cover PNG header and dimensions, scans text sources for common secret/PII patterns and host-specific paths, and refuses to treat a Kaggle `COMPLETE` state as semantic proof. A downloaded runtime result must pass both its outer contract and the duplicate-key-safe evidence normalizer before any card can be rendered.

The Version 7 source/metadata/vendor validator PASS establishes input and code
provenance only. Its explicit `DIAGNOSTIC_FAILURE` cannot satisfy the app's
runtime-artifact or semantic-evidence gates.

## Known limitations

A deterministic PASS demonstrates only that the authored evidence chain is internally complete. It does not prove source authenticity, official endorsement, currentness, native-language fluency, or real-world safety. Production use would require authenticated sources, native-speaker review, accessibility testing, adversarial evaluation, and field trials.
