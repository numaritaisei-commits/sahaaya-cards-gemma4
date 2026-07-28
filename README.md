# Sahaaya Cards

![Sahaaya Cards evidence-linked civic guidance cover](assets/cover.png)

Sahaaya Cards is an offline, multilingual civic-notice copilot for the **Voices of Bharat** track. This prototype turns each of two authored English-language civic notices into an exact-quote fact ledger and concise action cards in English, Hindi, and Tamil. A second Gemma 4 pass checks every visible claim, and deterministic Python validation rejects any result whose evidence chain is incomplete. It is a fixed, reproducible demonstration rather than a general file-upload product.

The prototype is intentionally narrow: it summarizes a supplied notice. It does not search the web, authenticate a notice, infer missing facts, contact authorities, or replace the current official notice.

## Current evidence status

The Gemma 4 candidate is **implemented and statically reviewable, but not yet runtime-proven**. The authored notebook, provenance gates, validators, renderer, synthetic fixtures, and a bounded one-generation runtime smoke exist. A Private, Internet-disabled Kaggle GPU run must still prove that the attached preset loads, both Gemma passes complete, and `demo_results.json` passes the downloaded validator. No generated card, runtime, PASS rate, or translation-quality result is claimed before that run.

## Why this matters

Time-sensitive notices are often long, monolingual, and difficult to scan. Translation alone is not enough: a fluent translation can quietly add a date, location, or instruction that was never present. Sahaaya Cards makes provenance visible. Each extracted fact must include an exact source quote, and every headline and action must cite fact IDs.

This design aims to help a reader see three things quickly: what the notice actually says, what action it supports, and what the system must not infer. It is a prototype for accessibility and evidence visibility, not an automated public-warning authority.

## Where Gemma 4 is essential

Gemma 4 Instruct 2B is the semantic engine, not a decorative wrapper:

1. **Generation pass.** Given the untrusted notice, Gemma 4 must extract an atomic fact ledger and draft exactly one English, Hindi, and Tamil card as strict JSON. Every fact carries an exact quote; every visible claim cites fact IDs.
2. **Verification pass.** Given the original notice plus the generated ledger and cards, the same Gemma 4 model performs a separate pass that evaluates each headline and action for entailment, cited evidence, and translation ambiguity.
3. **Fail-closed decision.** Deterministic code then enforces schema, exact-quote membership, ID integrity, three-language coverage, complete verifier coverage, and zero unsupported claims. Gemma provides multilingual interpretation; inspectable code provides structural guarantees.

The two prompts are separate calls with separate outputs. If generation fails its schema/evidence gate, verification is not run. If either stage fails, the result is retained as a failure record and is not rendered as an approved card.

## Kaggle runtime contract

- Competition: `build-with-gemma-ieee-cs-srmist-ktr`
- Official attached model: `keras/gemma4/keras/gemma4_instruct_2b/2`
- Model path: `/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_2b/2`
- Framework candidate: KerasHub 0.28.0 with Keras/JAX, `float16`, greedy sampling, and prompt stripping
- Notebook: Private
- Internet: Disabled
- Accelerator: one free Kaggle P100 GPU
- Inputs: the official Gemma 4 model, the competition source, two embedded fictional notices, and one private dataset containing the locally audited pure-Python KerasHub wheel
- Excluded: API keys, external services, personal data, shell commands, subprocesses, package installers, network fallbacks, and dynamic imports

The notebook does not blindly install the KerasHub wheel. Before import it checks the pinned archive SHA-256, size bounds, pure-Python tag, path safety, file types, complete wheel `RECORD`, and every recorded member digest. It extracts to a new local directory, verifies the extracted inventory again, and confirms that the imported KerasHub 0.28.0 package came from that directory. It separately inspects bounded model JSON and the attached inventory for `Gemma4Backbone`, `Gemma4CausalLM`, and the expected shards before loading `Gemma4CausalLM.from_preset(..., dtype="float16")`.

Both semantic passes use one 2048-token total-sequence budget so JAX compiles one decode shape. The prompt envelope and batched `{"prompts": [prompt]}` call follow the current official Gemma 4 and KerasHub examples; `validate_project.py` pins those exact source constructs. JAX preallocation is disabled and TensorFlow GPU growth is enabled before framework import. An atomic `runtime_journal.json` records only safe stage names, counts, status, and hashes; it leaves a diagnostic checkpoint even if model loading or compilation fails before `demo_results.json` is complete. Model JSON and model-returned JSON are bounded, and duplicate object keys are rejected.

These are implementation facts, not proof that the pending Kaggle run will succeed. There is no alternate model, network install, or unverified fallback.

## Reproducible artifact chain

- `sahaaya_cards_demo.ipynb` — self-contained two-notice Gemma 4 candidate
- `kernel-metadata.example.json` — Private, Internet-off, free-GPU template;
  replace `YOUR_KAGGLE_USERNAME` only in your own Kaggle copy
- `dataset-metadata.example.json` — Private wheel-Dataset metadata template
- `fixtures/` — the same two fictional notices in machine-readable form
- `validate_project.py` — fail-closed static and runtime-artifact validator
- `build_demo.py` — standard-library-only offline HTML renderer; refuses absent or failing runtime artifacts
- `tests/` — validator and renderer refusal/escaping tests
- `runtime_smoke/` — one bounded generation proving the exact wheel/model/GPU contract before the full run
- `tools/` and `dependencies/` — wheel audit implementation, audit result, and upstream license; the wheel itself is excluded
- `schemas/` — machine-readable private runtime-artifact contract
- `DEPENDENCIES.md` and `SECURITY_AND_PRIVACY.md` — source provenance and public/private boundaries
- `MANIFEST.sha256` — exact public source inventory and hashes

Run the local checks without loading the model or importing the wheel:

```text
python3 validate_project.py
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s runtime_smoke/tests -v
python3 runtime_smoke/validate_smoke.py \
  runtime_smoke/gemma4_smoke.py runtime_smoke/kernel-metadata.example.json
```

Validate a Kaggle-produced `demo_results.json` directly from its download directory;
it does not need to be copied into the reviewed source tree:

```text
python3 validate_project.py --runtime /path/to/downloaded/demo_results.json
```

This command performs both the outer schema/configuration checks and the
renderer’s duplicate-key-safe deep evidence normalization: trusted fixture
hashes, exact quotes, fact IDs, card coverage, verifier coverage, and timings.
Kaggle kernel `COMPLETE` alone is never treated as a card PASS.

Only after that command reports `runtime_artifact=PASS` and
`runtime_semantic_evidence=PASS`, build the self-contained HTML demo:

```text
python3 build_demo.py --input /path/to/downloaded/demo_results.json
```

The renderer calls `validate_runtime_artifact(..., require_pass=True)`, rechecks fixture hashes, ledger quotes, card evidence, and verifier coverage, then renders only a fixed field whitelist. Every runtime-derived value is escaped. The page has an offline Content Security Policy and contains no JavaScript, remote asset, form, analytics, or network request. If the runtime artifact is absent or invalid, no page is written.

## Responsible boundaries

A deterministic PASS would mean that this prototype found a complete evidence chain under its authored checks. It would not mean that an authority endorsed the card, that the source was authentic/current, or that every translation nuance was correct. Critical instructions still require comparison with the current official notice. Real deployment would require authenticated source ingestion, native-speaker review, accessibility testing, adversarial evaluation, and field trials with affected communities.

## License

Project code and documentation are released under Apache License 2.0. The attached Gemma model remains governed by its upstream terms and is not redistributed here.
