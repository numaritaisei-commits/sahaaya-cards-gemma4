#!/usr/bin/env python3
"""Render a validated Sahaaya Cards artifact as a self-contained offline page."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import validate_project as validator


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "demo_results.json"
DEFAULT_OUTPUT = ROOT / "demo" / "index.html"
MAX_ARTIFACT_BYTES = 4_000_000
MAX_RAW_FINAL_UTF8_BYTES = 131_072
LANGUAGE_LABELS = {"en": "English", "hi": "हिन्दी", "ta": "தமிழ்"}
ALLOWED_FACT_KINDS = {
    "time",
    "place",
    "action",
    "contact",
    "constraint",
    "service",
    "warning",
}
FACT_ID_RE = re.compile(r"^F[1-9][0-9]*$")


class DemoRefusal(ValueError):
    """Raised when untrusted runtime content is not safe to render."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DemoRefusal(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise DemoRefusal(f"non-finite JSON number is forbidden: {token}")


def _exact_object(value: Any, field: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DemoRefusal(f"field inventory mismatch: {field}")
    return value


def _parse_raw_object(value: Any, field: str) -> dict[str, Any]:
    raw = _text(value, field, maximum=MAX_RAW_FINAL_UTF8_BYTES)
    if len(raw.encode("utf-8")) > MAX_RAW_FINAL_UTF8_BYTES:
        raise DemoRefusal(f"raw final answer exceeds byte limit: {field}")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite,
        )
    except (json.JSONDecodeError, DemoRefusal) as exc:
        raise DemoRefusal(f"raw final answer is not strict JSON: {field}") from exc
    if not isinstance(parsed, dict):
        raise DemoRefusal(f"raw final answer must be a JSON object: {field}")
    return parsed


def _canonical_json(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DemoRefusal(f"non-canonical JSON value: {field}") from exc


def _text(value: Any, field: str, *, maximum: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DemoRefusal(f"invalid text field: {field}")
    if len(value) > maximum:
        raise DemoRefusal(f"text field exceeds limit: {field}")
    return value


def _string_list(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum_items: int = 20,
    maximum_chars: int = 1000,
) -> list[str]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum_items:
        raise DemoRefusal(f"invalid list field: {field}")
    return [
        _text(item, f"{field}[{index}]", maximum=maximum_chars)
        for index, item in enumerate(value)
    ]


def _fact_id_list(
    value: Any, field: str, known: set[str], *, minimum: int = 1
) -> list[str]:
    values = _string_list(
        value, field, minimum=minimum, maximum_items=30, maximum_chars=12
    )
    if len(values) != len(set(values)):
        raise DemoRefusal(f"duplicate fact ID in {field}")
    if not all(FACT_ID_RE.fullmatch(item) and item in known for item in values):
        raise DemoRefusal(f"unknown or malformed fact ID in {field}")
    return values


def _fixture_map() -> dict[str, dict[str, Any]]:
    fixtures: dict[str, dict[str, Any]] = {}
    for path in validator.FIXTURE_PATHS:
        if not path.is_file() or path.is_symlink():
            raise DemoRefusal("trusted synthetic fixture is missing or is a symlink")
        fixture = validator.load_json(path)
        fixture_errors = validator.validate_fixture(fixture, path)
        if fixture_errors:
            raise DemoRefusal("trusted synthetic fixture failed validation")
        fixtures[fixture["notice_id"]] = fixture
    if set(fixtures) != validator.EXPECTED_NOTICE_IDS:
        raise DemoRefusal("trusted synthetic fixture set is incomplete")
    return fixtures


def _normalize_notice(
    result: dict[str, Any], fixture: dict[str, Any]
) -> dict[str, Any]:
    _exact_object(
        result,
        "notice",
        {
            "notice_id",
            "source_sha256",
            "prompts",
            "raw_final_answers",
            "parsed",
            "token_budgets",
            "timing_seconds",
            "validation",
        },
    )
    notice_id = _text(result.get("notice_id"), "notice_id", maximum=40)
    expected_source_hash = hashlib.sha256(
        json.dumps(fixture, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if result.get("source_sha256") != expected_source_hash:
        raise DemoRefusal("source hash does not match the trusted synthetic fixture")
    prompts = _exact_object(
        result.get("prompts"),
        "prompts",
        {"generator", "verifier"},
    )
    for role in ("generator", "verifier"):
        _text(prompts[role], f"prompts.{role}", maximum=MAX_RAW_FINAL_UTF8_BYTES)

    validation = _exact_object(
        result.get("validation"),
        "validation",
        {"passed", "errors"},
    )
    if validation.get("passed") is not True or validation.get("errors") != []:
        raise DemoRefusal("notice did not pass the deterministic runtime gate")

    parsed = _exact_object(
        result.get("parsed"),
        "parsed",
        {"generator", "verifier"},
    )
    bundle = parsed.get("generator")
    verification = parsed.get("verifier")
    if not isinstance(bundle, dict) or not isinstance(verification, dict):
        raise DemoRefusal("parsed generator or verifier result is missing")
    _exact_object(
        bundle,
        "parsed.generator",
        {"notice_id", "fact_ledger", "cards", "uncertainties"},
    )
    _exact_object(
        verification,
        "parsed.verifier",
        {
            "notice_id",
            "verdict",
            "checks",
            "unsupported_claims",
            "language_warnings",
            "safety_note",
        },
    )
    raw_answers = _exact_object(
        result.get("raw_final_answers"),
        "raw_final_answers",
        {"generator", "verifier"},
    )
    for role, normalized in (("generator", bundle), ("verifier", verification)):
        raw_parsed = _parse_raw_object(
            raw_answers[role],
            f"raw_final_answers.{role}",
        )
        if _canonical_json(raw_parsed, role) != _canonical_json(normalized, role):
            raise DemoRefusal(f"raw and parsed {role} answers differ")
    if bundle.get("notice_id") != notice_id or verification.get("notice_id") != notice_id:
        raise DemoRefusal("parsed notice IDs do not match")

    ledger_value = bundle.get("fact_ledger")
    if not isinstance(ledger_value, list) or not 1 <= len(ledger_value) <= 6:
        raise DemoRefusal("fact ledger must contain one to six facts")
    source_text = "\n".join(
        [fixture["issuer"], fixture["issued_at"], fixture["title"], fixture["body"]]
    )
    facts: list[dict[str, str]] = []
    known_ids: set[str] = set()
    expected_claims: dict[str, set[str]] = {}
    for index, raw_fact in enumerate(ledger_value):
        _exact_object(
            raw_fact,
            f"facts[{index}]",
            {"fact_id", "kind", "value", "source_quote"},
        )
        fact_id = _text(raw_fact.get("fact_id"), f"facts[{index}].fact_id", maximum=12)
        if not FACT_ID_RE.fullmatch(fact_id) or fact_id in known_ids:
            raise DemoRefusal("fact IDs must be unique F-number identifiers")
        kind = _text(raw_fact.get("kind"), f"facts[{index}].kind", maximum=24)
        if kind not in ALLOWED_FACT_KINDS:
            raise DemoRefusal("fact kind is not allowed")
        value = _text(raw_fact.get("value"), f"facts[{index}].value", maximum=600)
        quote = _text(raw_fact.get("source_quote"), f"facts[{index}].source_quote", maximum=800)
        if quote not in source_text:
            raise DemoRefusal("fact quote is not an exact synthetic-source substring")
        known_ids.add(fact_id)
        facts.append({"fact_id": fact_id, "kind": kind, "value": value, "quote": quote})
        expected_claims[f"fact_ledger[{index}].value"] = {fact_id}

    cards_value = bundle.get("cards")
    if not isinstance(cards_value, list) or len(cards_value) != 3:
        raise DemoRefusal("card bundle must contain exactly three cards")
    cards_by_language: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()
    for card_index, raw_card in enumerate(cards_value):
        _exact_object(
            raw_card,
            f"cards[{card_index}]",
            {"language", "headline", "source_fact_ids", "actions", "do_not_infer"},
        )
        language = _text(raw_card.get("language"), f"cards[{card_index}].language", maximum=2)
        if language not in LANGUAGE_LABELS or language in cards_by_language:
            raise DemoRefusal("card languages must be exactly en, hi, and ta")
        headline = _text(raw_card.get("headline"), f"cards[{card_index}].headline", maximum=140)
        headline_ids = _fact_id_list(
            raw_card.get("source_fact_ids"),
            f"cards[{card_index}].source_fact_ids",
            known_ids,
        )
        used_ids.update(headline_ids)
        expected_claims[f"cards[{card_index}].headline"] = set(headline_ids)
        actions_value = raw_card.get("actions")
        if not isinstance(actions_value, list) or len(actions_value) != 1:
            raise DemoRefusal("each card must contain exactly one action")
        actions: list[dict[str, Any]] = []
        for action_index, raw_action in enumerate(actions_value):
            _exact_object(
                raw_action,
                f"cards[{card_index}].actions[{action_index}]",
                {"text", "fact_ids"},
            )
            action_text = _text(
                raw_action.get("text"),
                f"cards[{card_index}].actions[{action_index}].text",
                maximum=260,
            )
            if len(action_text.split()) > 35:
                raise DemoRefusal("card action exceeds 35 whitespace-delimited words")
            action_ids = _fact_id_list(
                raw_action.get("fact_ids"),
                f"cards[{card_index}].actions[{action_index}].fact_ids",
                known_ids,
            )
            used_ids.update(action_ids)
            expected_claims[
                f"cards[{card_index}].actions[{action_index}].text"
            ] = set(action_ids)
            actions.append({"text": action_text, "fact_ids": action_ids})
        limitations = _string_list(
            raw_card.get("do_not_infer"),
            f"cards[{card_index}].do_not_infer",
            minimum=1,
            maximum_items=8,
            maximum_chars=500,
        )
        for limit_index in range(len(limitations)):
            expected_claims[
                f"cards[{card_index}].do_not_infer[{limit_index}]"
            ] = set()
        cards_by_language[language] = {
            "language": language,
            "headline": headline,
            "source_fact_ids": headline_ids,
            "actions": actions,
            "do_not_infer": limitations,
        }
    if set(cards_by_language) != set(LANGUAGE_LABELS):
        raise DemoRefusal("card languages must be exactly en, hi, and ta")
    if used_ids != known_ids:
        raise DemoRefusal("every ledger fact must be used by at least one card claim")

    if verification.get("verdict") != "PASS":
        raise DemoRefusal("verifier verdict is not PASS")
    if verification.get("unsupported_claims") != []:
        raise DemoRefusal("verifier reported unsupported claims")
    if verification.get("language_warnings") != []:
        raise DemoRefusal("verifier reported language warnings")
    safety_note = _text(verification.get("safety_note"), "verifier.safety_note", maximum=800)
    checks_value = verification.get("checks")
    if not isinstance(checks_value, list) or len(checks_value) != len(expected_claims):
        raise DemoRefusal("verifier check coverage is incomplete")
    observed_paths: set[str] = set()
    for index, raw_check in enumerate(checks_value):
        _exact_object(
            raw_check,
            f"checks[{index}]",
            {"claim_path", "supported", "fact_ids", "explanation"},
        )
        claim_path = _text(raw_check.get("claim_path"), f"checks[{index}].claim_path", maximum=100)
        if claim_path not in expected_claims or claim_path in observed_paths:
            raise DemoRefusal("verifier claim path is unknown or duplicated")
        if raw_check.get("supported") is not True:
            raise DemoRefusal("verifier check is not supported")
        fact_ids = _fact_id_list(
            raw_check.get("fact_ids"),
            f"checks[{index}].fact_ids",
            known_ids,
            minimum=0,
        )
        if set(fact_ids) != expected_claims[claim_path]:
            raise DemoRefusal("verifier fact IDs do not match displayed evidence")
        _text(raw_check.get("explanation"), f"checks[{index}].explanation", maximum=800)
        observed_paths.add(claim_path)
    if observed_paths != set(expected_claims):
        raise DemoRefusal("verifier did not cover every visible claim")

    uncertainties = _string_list(
        bundle.get("uncertainties"),
        "uncertainties",
        minimum=0,
        maximum_items=12,
        maximum_chars=800,
    )
    budgets = _exact_object(
        result.get("token_budgets"),
        "token_budgets",
        {"generator", "verifier"},
    )
    budget_keys = {
        "completion_budget_tokens",
        "max_length",
        "minimum_completion_tokens",
        "prompt_tokens",
    }
    for role in ("generator", "verifier"):
        _exact_object(budgets.get(role), f"token_budgets.{role}", budget_keys)

    timing = _exact_object(
        result.get("timing_seconds"),
        "timing_seconds",
        {"generator", "verifier", "total"},
    )
    normalized_timing: dict[str, float] = {}
    for key in ("generator", "verifier", "total"):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 86_400:
            raise DemoRefusal("timing value is invalid")
        normalized_timing[key] = float(value)

    return {
        "notice_id": notice_id,
        "fixture": {
            "issuer": _text(fixture["issuer"], "fixture.issuer", maximum=200),
            "issued_at": _text(fixture["issued_at"], "fixture.issued_at", maximum=50),
            "title": _text(fixture["title"], "fixture.title", maximum=200),
            "body": _text(fixture["body"], "fixture.body", maximum=4000),
        },
        "source_sha256": expected_source_hash,
        "facts": facts,
        "cards": [cards_by_language[language] for language in LANGUAGE_LABELS],
        "uncertainties": uncertainties,
        "verifier": {"verdict": "PASS", "safety_note": safety_note, "check_count": len(checks_value)},
        "timing": normalized_timing,
    }


def _load_and_normalize(source: Path) -> tuple[dict[str, Any], str]:
    if not source.is_file() or source.is_symlink():
        raise DemoRefusal("validated demo_results.json is absent or unsafe")
    if source.stat().st_size > MAX_ARTIFACT_BYTES:
        raise DemoRefusal("runtime artifact exceeds the offline demo size limit")
    raw_bytes = source.read_bytes()
    try:
        artifact = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, DemoRefusal) as exc:
        raise DemoRefusal("runtime artifact is not valid UTF-8 JSON") from exc
    runtime_errors = validator.validate_runtime_artifact(artifact, require_pass=True)
    if runtime_errors:
        raise DemoRefusal(
            f"runtime artifact failed the authoritative validator ({len(runtime_errors)} findings)"
        )
    _exact_object(
        artifact,
        "runtime artifact",
        {
            "schema_version",
            "project",
            "generated_at_utc",
            "status",
            "model_ref",
            "model_path",
            "run_configuration",
            "runtime_provenance",
            "safety_limitations",
            "failures",
            "notices",
        },
    )
    fixtures = _fixture_map()
    runtime_notices = artifact["notices"]
    notices_by_id = {result["notice_id"]: result for result in runtime_notices}
    if len(runtime_notices) != len(validator.EXPECTED_NOTICE_IDS) or len(notices_by_id) != len(
        validator.EXPECTED_NOTICE_IDS
    ):
        raise DemoRefusal("runtime artifact must contain each synthetic notice exactly once")
    notices = [
        _normalize_notice(notices_by_id[notice_id], fixtures[notice_id])
        for notice_id in sorted(validator.EXPECTED_NOTICE_IDS)
    ]
    limitations = _string_list(
        artifact.get("safety_limitations"),
        "safety_limitations",
        minimum=4,
        maximum_items=12,
        maximum_chars=1000,
    )
    generated_at = _text(artifact.get("generated_at_utc"), "generated_at_utc", maximum=80)
    return (
        {
            "project": "Sahaaya Cards",
            "competition": validator.COMPETITION,
            "track": "Voices of Bharat",
            "generated_at": generated_at,
            "model_ref": validator.MODEL_REF,
            "configuration": {
                "framework": "KerasHub",
                "sampler": "greedy",
                "prompt_output": "prompt stripped",
                "internet": "disabled",
                "external_apis": "none",
            },
            "limitations": limitations,
            "notices": notices,
        },
        hashlib.sha256(raw_bytes).hexdigest(),
    )


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _render_html(view: dict[str, Any], artifact_hash: str) -> str:
    sections: list[str] = []
    for notice in view["notices"]:
        fixture = notice["fixture"]
        facts = "".join(
            "<tr>"
            f"<td><code>{_escape(fact['fact_id'])}</code></td>"
            f"<td>{_escape(fact['kind'])}</td>"
            f"<td>{_escape(fact['value'])}</td>"
            f"<td><q>{_escape(fact['quote'])}</q></td>"
            "</tr>"
            for fact in notice["facts"]
        )
        cards: list[str] = []
        for card in notice["cards"]:
            actions = "".join(
                "<li>"
                f"<span>{_escape(action['text'])}</span> "
                f"<code>{_escape(', '.join(action['fact_ids']))}</code>"
                "</li>"
                for action in card["actions"]
            )
            boundaries = "".join(
                f"<li>{_escape(item)}</li>" for item in card["do_not_infer"]
            )
            cards.append(
                "<article class=\"card\">"
                f"<p class=\"language\">{_escape(LANGUAGE_LABELS[card['language']])}</p>"
                f"<h4>{_escape(card['headline'])}</h4>"
                f"<p class=\"evidence\">Headline evidence: <code>{_escape(', '.join(card['source_fact_ids']))}</code></p>"
                f"<ol>{actions}</ol>"
                "<details><summary>Do not infer</summary>"
                f"<ul>{boundaries}</ul></details>"
                "</article>"
            )
        uncertainty_html = "".join(
            f"<li>{_escape(item)}</li>" for item in notice["uncertainties"]
        ) or "<li>None recorded</li>"
        sections.append(
            "<section class=\"notice\">"
            f"<p class=\"eyebrow\">{_escape(notice['notice_id'])}</p>"
            f"<h2>{_escape(fixture['title'])}</h2>"
            f"<p class=\"meta\">Issued by {_escape(fixture['issuer'])} · {_escape(fixture['issued_at'])}</p>"
            f"<div class=\"source\"><h3>Synthetic source notice</h3><p>{_escape(fixture['body'])}</p></div>"
            "<h3>Grounded fact ledger</h3>"
            "<div class=\"table-wrap\"><table><thead><tr><th>ID</th><th>Kind</th><th>Normalized fact</th><th>Exact source quote</th></tr></thead>"
            f"<tbody>{facts}</tbody></table></div>"
            "<h3>Action cards</h3>"
            f"<div class=\"cards\">{''.join(cards)}</div>"
            "<div class=\"verification\">"
            f"<strong>Verifier {_escape(notice['verifier']['verdict'])}</strong> · "
            f"{notice['verifier']['check_count']} visible claims checked"
            f"<p>{_escape(notice['verifier']['safety_note'])}</p></div>"
            f"<details><summary>Uncertainties</summary><ul>{uncertainty_html}</ul></details>"
            f"<p class=\"hash\">Source SHA-256: <code>{_escape(notice['source_sha256'])}</code></p>"
            "<p class=\"timing\">Runtime: "
            f"generator {notice['timing']['generator']:.3f}s · verifier {notice['timing']['verifier']:.3f}s · "
            f"total {notice['timing']['total']:.3f}s</p>"
            "</section>"
        )

    limitations = "".join(f"<li>{_escape(item)}</li>" for item in view["limitations"])
    config = view["configuration"]
    return "<!doctype html>\n" + f"""<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; font-src 'none'; connect-src 'none'; media-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Sahaaya Cards — offline demo</title>
<style>
:root {{ color-scheme: light; --ink:#17242f; --muted:#5d6b73; --paper:#f5f3ec; --card:#fffdf7; --line:#d8d1c4; --teal:#0b6b66; --saffron:#d67821; --pass:#17653a; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 system-ui,-apple-system,sans-serif; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:auto; padding:48px 0 80px; }}
header {{ padding:34px; border:1px solid var(--line); background:linear-gradient(135deg,#fffdf7,#e5f2ed); border-radius:24px; }}
h1 {{ font-size:clamp(2.2rem,6vw,4.8rem); line-height:.95; margin:.25rem 0 1rem; letter-spacing:-.055em; }}
h2 {{ font-size:2rem; margin:.15rem 0; }} h3 {{ margin-top:1.8rem; }} h4 {{ font-size:1.2rem; margin:.2rem 0 .6rem; }}
.eyebrow,.language {{ color:var(--teal); font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
.badges {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:20px; }} .badge {{ border:1px solid var(--teal); border-radius:999px; padding:5px 10px; font-size:.86rem; }}
.notice {{ margin-top:28px; padding:30px; background:var(--card); border:1px solid var(--line); border-radius:22px; box-shadow:0 12px 35px #26372a12; }}
.meta,.hash,.timing,.evidence {{ color:var(--muted); font-size:.9rem; }} .source {{ padding:18px; border-left:5px solid var(--saffron); background:#fff8ea; }}
.cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }} .card {{ padding:18px; border:1px solid var(--line); border-radius:16px; background:white; }}
.card ol {{ padding-left:1.35rem; }} .card li {{ margin:.65rem 0; }} code {{ overflow-wrap:anywhere; color:#064f4a; }}
.table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; min-width:760px; }} th,td {{ text-align:left; vertical-align:top; padding:10px; border-bottom:1px solid var(--line); }} th {{ color:var(--muted); font-size:.82rem; text-transform:uppercase; }}
.verification {{ margin-top:18px; padding:16px 18px; border:1px solid #9bc9aa; border-radius:14px; background:#edf9f0; color:var(--pass); }}
.provenance,.limitations {{ margin-top:28px; padding:24px; border:1px solid var(--line); border-radius:18px; background:#fffdf7; }}
footer {{ margin-top:30px; color:var(--muted); }}
@media (max-width:850px) {{ .cards {{ grid-template-columns:1fr; }} .notice,header {{ padding:20px; }} }}
@media print {{ body {{ background:white; }} .notice {{ break-inside:avoid; box-shadow:none; }} }}
</style>
</head>
<body><main>
<header>
<p class="eyebrow">Voices of Bharat · Offline civic access</p>
<h1>Sahaaya Cards</h1>
<p>Evidence-linked English, Hindi, and Tamil action cards from a local civic notice—generated and verified offline with Gemma 4.</p>
<div class="badges"><span class="badge">Private Kaggle notebook</span><span class="badge">Internet disabled</span><span class="badge">No external APIs</span><span class="badge">Fail-closed validation</span></div>
</header>
{''.join(sections)}
<section class="provenance"><h2>Runtime &amp; provenance</h2>
<dl><dt>Competition</dt><dd>{_escape(view['competition'])}</dd><dt>Track</dt><dd>{_escape(view['track'])}</dd>
<dt>Model source</dt><dd><code>{_escape(view['model_ref'])}</code></dd><dt>Generated at</dt><dd>{_escape(view['generated_at'])}</dd>
<dt>Configuration</dt><dd>{_escape(config['framework'])}; sampler {_escape(config['sampler'])}; {_escape(config['prompt_output'])}; Internet {_escape(config['internet'])}; external APIs {_escape(config['external_apis'])}</dd>
<dt>Runtime artifact SHA-256</dt><dd><code>{_escape(artifact_hash)}</code></dd></dl></section>
<section class="limitations"><h2>Safety limitations</h2><ul>{limitations}</ul></section>
<footer><p>This self-contained page contains no script, remote font, image, analytics, form, or network request. A verifier PASS confirms the recorded evidence chain; it is not an official endorsement.</p></footer>
</main></body></html>"""


def render_demo(source: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT) -> Path:
    view, artifact_hash = _load_and_normalize(source)
    document = _render_html(view, artifact_hash)
    output_parent = output.parent
    if output.exists() and output.is_symlink():
        raise DemoRefusal("output path must not be a symlink")
    if output_parent.exists() and output_parent.is_symlink():
        raise DemoRefusal("output directory must not be a symlink")
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".sahaaya-demo-",
            suffix=".tmp",
            dir=output_parent,
            delete=False,
        ) as handle:
            handle.write(document)
            temporary_path = Path(handle.name)
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        output = render_demo(args.input, args.output)
    except (DemoRefusal, OSError) as exc:
        print(f"HOLD: renderer refused input ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"GO: wrote offline demo to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
