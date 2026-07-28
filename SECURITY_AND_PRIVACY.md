# Security and privacy boundary

Sahaaya Cards is a fixed research prototype over two fictional notices. It is not a general upload service and does not authenticate, browse for, or contact an authority.

## Runtime boundary

- Private Kaggle notebook, free GPU, Internet disabled.
- Official Gemma 4 Kaggle Model plus one locally audited pure-Python KerasHub wheel Dataset.
- No API key, Cookie, email address, phone number, user document, or other personal data.
- No package installer, network fallback, subprocess, shell command, downloaded native executable, `eval`, or `exec`.
- Prompt-delimited notices and candidate responses are untrusted data, never instructions.

## Publication boundary

This repository intentionally excludes credentials, private Kaggle metadata, wheel binaries, model weights, generated prompts and responses, runtime journals, raw result artifacts, video files, and local filesystem paths. `kernel-metadata.example.json` and `dataset-metadata.example.json` contain owner placeholders only.

The public validator uses an exact source allowlist, rejects symlinks, verifies the integrity manifest, checks the cover PNG header and dimensions, scans text sources for common secret/PII patterns and host-specific paths, and refuses to treat a Kaggle `COMPLETE` state as semantic proof. A downloaded runtime result must pass both its outer contract and the duplicate-key-safe evidence normalizer before any card can be rendered.

## Known limitations

A deterministic PASS demonstrates only that the authored evidence chain is internally complete. It does not prove source authenticity, official endorsement, currentness, native-language fluency, or real-world safety. Production use would require authenticated sources, native-speaker review, accessibility testing, adversarial evaluation, and field trials.

