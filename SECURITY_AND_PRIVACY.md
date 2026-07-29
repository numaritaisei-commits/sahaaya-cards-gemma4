# Security and privacy boundary

Sahaaya Cards is a fixed research prototype over two fictional notices. It is not a general upload service and does not authenticate, browse for, or contact an authority.

## Runtime boundary

- The Keras/Torch two-pass app uses a Private Kaggle notebook, free T4 x2, and
  Internet disabled. Its V4-V6 attempts remained incomplete, so this repository
  claims no successful app run.
- The JAX structure diagnostic Version 4 used an exact Private,
  Internet-disabled T4 x2 run and completed in 44.350 seconds with
  `STRUCTURE_SHARDING_PASS`. It used `load_weights=False`, so it validates model
  structure and sharding only.
- The JAX weighted diagnostic Version 1 used the same approved inputs and
  layout. Its exact source, metadata, and sanitized outputs passed the
  fail-closed validator, but it ended after 176.725 seconds as
  `DIAGNOSTIC_FAILURE` / `VALUE` at `model_load_started` with
  `weights_loaded=false` and `generation_attempted=false`. Cleanup completed;
  full checkpoint restoration and application execution did not.
- JAX weighted generation Version 3 feature-column sharded the per-layer
  embedding and completed in 498.684 seconds with
  `WEIGHTED_GENERATION_PASS`. It restored every official checkpoint weight and
  completed one bounded greedy generation on T4 x2. This was not an application
  pass: no civic-notice prompt, card, translation, verifier, or app artifact was
  produced.
- Official Gemma 4 Kaggle Model plus one locally audited pure-Python KerasHub wheel Dataset.
- No API key, Cookie, email address, phone number, user document, or other personal data.
- No package installer, network fallback, subprocess, shell command, downloaded native executable, `eval`, or `exec`.
- Prompt-delimited notices and candidate responses are untrusted data, never instructions.

## Publication boundary

This repository intentionally excludes credentials, private Kaggle metadata,
wheel binaries, model weights, generated prompts and responses, runtime
journals, raw result artifacts, model/runtime videos, and local filesystem paths.
Only bounded, validated JAX outcomes are summarized here; generated text and
its hash remain unpublished. `kernel-metadata.example.json` and
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

The JAX sharding PASS establishes structure and layout feasibility. The weighted
artifacts preserve both the earlier checkpoint-restoration failure and the
later bounded generation success. Neither satisfies the app's runtime-artifact
or semantic-evidence gates.

## Known limitations

A deterministic PASS demonstrates only that the authored evidence chain is internally complete. It does not prove source authenticity, official endorsement, currentness, native-language fluency, or real-world safety. Production use would require authenticated sources, native-speaker review, accessibility testing, adversarial evaluation, and field trials.
