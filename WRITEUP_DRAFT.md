# Sahaaya Cards: evidence-linked civic guidance with Gemma 4

**Subtitle:** An offline, multilingual notice-card design with an explicit source-to-claim chain

## The problem

A civic notice can be public yet hard to use. It may be long, available in one
language, and full of dates or conditions that are difficult to scan during a
disruption. A fluent summary can also sound authoritative while quietly adding
an unsupported place, time, or instruction.

Sahaaya Cards is a fixed prototype for the **Voices of Bharat** track. Its
reviewed implementation is designed to transform two fictional English notices
into concise English, Hindi, and Tamil cards while retaining visible evidence
for every headline and action. It is not a general upload product or an
automated public-warning authority.

## Evidence before presentation

The first Gemma 4 pass is designed to return strict JSON containing no more than
six atomic facts, an exact source quote for each fact, and one card per language.
Every displayed headline and action must cite fact IDs. A separate Gemma 4 pass
then evaluates each proposed claim for entailment, cited evidence, and
translation ambiguity.

Deterministic Python is designed to fail closed unless:

- the result contains exactly one English, Hindi, and Tamil card;
- every ledger quote is an exact substring of the supplied notice;
- every cited fact exists and every fact is used;
- every fact value, headline, action, and displayed boundary has one verifier check;
- verifier evidence IDs match the card evidence IDs; and
- no unsupported claim or language warning remains.

If generation fails its structural gate, verification is not run. A preflight,
load, generation, or validation error writes only a bounded non-PASS diagnostic.
No partial result may be rendered as an approved card.

## Why Gemma 4 is substantive

The code uses the official Kaggle preset
`keras/gemma4/keras/gemma4_instruct_2b/2` through
`Gemma4CausalLM`. Gemma 4 is assigned both grounded multilingual abstraction
and semantic cross-checking; deterministic code supplies the explicit evidence
contract. The repository does not substitute canned cards for a successful
model run.

The notebook is output-free but preserves the complete official-model loading,
generation, verification, bounded-artifact, and refusal paths for review.

## Offline and inspectable by design

The reviewed route is Private and Internet-disabled. It uses the official Gemma
4 model plus a pinned KerasHub 0.28.0 pure-Python wheel. Before import, the
notebook verifies the wheel archive hash, path and size bounds, pure-Python tag,
complete `RECORD`, and every recorded member digest. It separately checks the
bounded model configuration and expected weight shards. It uses Keras with the
Torch backend, plain `float16`, greedy decoding, prompt stripping, and no
external API.

The notices are clearly fictional. The code contains no credential, personal
record, package installer, network fallback, subprocess, dynamic import,
telemetry, model weight, or wheel binary. Generated JSON would be treated as
untrusted data, reject duplicate keys, and remain hidden from notebook logs.

## Measured runtime evidence — bounded generation, app incomplete

This submission does **not** claim that Gemma 4 generated or verified a card.
The Keras/Torch two-pass app had three Private, Internet-disabled Kaggle T4 x2
development attempts:

- V4 ended during direct GPU model loading with an out-of-memory error.
- V5 attempted CPU staging and was killed after host memory exhaustion.
- V6 also failed to reach validated end-to-end generation under the available memory.

Two separate JAX model-parallel diagnostics narrow the runtime boundary without
establishing an application pass. The JAX structure diagnostic Version 4
completed in 44.350 seconds with `STRUCTURE_SHARDING_PASS`: 1,951 distinct
variables, 248 layout-matched variables sharded across two T4 GPUs,
10,822,965,702 global variable bytes, 6,208,838,086 estimated per-device bytes,
and 127 bounded collisions with maximum kind `SAFE_SHAPE_VARIANT`. It used
`load_weights=False`; this proves only that the reviewed model structure and
layout instantiated and sharded.

The JAX weighted diagnostic Version 1 used the same pinned model, audited
wheel, runtime versions, and layout. Its exact source, server metadata, and
sanitized outputs passed the fail-closed validator, but the run ended after
176.725 seconds as `DIAGNOSTIC_FAILURE` / `VALUE` at `model_load_started`.
`official_checkpoint_restored=false`, `weights_loaded=false`, and
`generation_attempted=false`; cleanup completed. Full checkpoint restoration
did not complete, and no generation, fact ledger, multilingual card,
verification pass, or app artifact was produced.

JAX weighted generation Version 3 tested the corrective hypothesis rather than
hiding that failure. It feature-column sharded the `(262144, 8960)` per-layer
embedding across axis 1. In an Internet-disabled T4 x2 run, every official
checkpoint weight was restored, full source and target tensor digests matched,
and one non-empty greedy generation completed. The fail-closed terminal status
was `WEIGHTED_GENERATION_PASS`: 498.684 seconds total, including 427.421 seconds
to load and 43.298 seconds to generate. Exact source and sanitized outputs
passed the dedicated validator.

[The validated source and bounded outputs are public on Kaggle](https://www.kaggle.com/code/numaritaisei/sahaaya-cards-gemma4-jax-host-half-restore).

That success is not an application pass. Version 3 did not run either Sahaaya
Cards prompt, produce a fact ledger or multilingual card, execute the verifier,
or create `demo_results.json`. Therefore there is still no app PASS rate,
translation-quality score, validated model-output video, or measured impact to
report. The hand-authored
`ILLUSTRATIVE_OUTPUT.md` demonstrates only the proposed interface and is
prominently labeled **not model-generated**.

The [114-second silent fixture walkthrough](assets/sahaaya-cards-fixture-prototype.mp4)
uses the same two fictional notices and keeps that evidence boundary visible in
every scene. It is hand-authored illustrative media, not model output, a
validated translation, or proof of a successful runtime.

The validated JAX structure and weighted generation evidence establish a
working official-checkpoint inference route. They do not convert one bounded
generation into application success. Completing the reviewed app still
requires both model passes and the deterministic card-evidence gate.

## Value and responsible boundaries

The contribution is the inspectable source-to-claim contract: exact quotes,
fact IDs, independent semantic checks, complete-coverage enforcement, bounded
outputs, and refusal to turn incomplete evidence into a polished result.

Even a future deterministic PASS would establish only that these authored
checks found an internally complete evidence chain. It would not authenticate a
notice, provide official endorsement, establish currentness, or prove native
language quality. Real deployment would additionally require authenticated
sources, native-speaker review, accessibility testing, adversarial evaluation,
and field trials with affected communities.
