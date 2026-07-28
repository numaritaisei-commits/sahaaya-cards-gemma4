#!/usr/bin/env python3
"""Fail-closed, standard-library validation for the Sahaaya Cards prototype."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
NOTEBOOK_NAME = "sahaaya_cards_demo.ipynb"
MODEL_REF = "keras/gemma4/keras/gemma4_instruct_2b/2"
MODEL_PATH = "/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_2b/2"
WHEEL_DATASET_REF = "YOUR_KAGGLE_USERNAME/verified-keras-hub-028-gemma4"
KERNEL_METADATA_NAME = "kernel-metadata.example.json"
WHEEL_SHA256 = "a28cb601f7fffb7f28add1bae8110459fc3ac7d9e2159453dfbda9e97271fc87"
FORBIDDEN_NOTEBOOK_TOKENS = (
    "Gemma3CausalLM",
    "gemma3_instruct",
    "AutoProcessor",
    "AutoModelForCausalLM",
    "AutoModelForMultimodalLM",
    "apply_chat_template",
    "import torch",
    "from transformers",
    "pip install",
    "!pip",
)
COMPETITION = "build-with-gemma-ieee-cs-srmist-ktr"
FIXTURE_PATHS = (
    ROOT / "fixtures" / "synthetic_flood_notice.json",
    ROOT / "fixtures" / "synthetic_water_notice.json",
)
EXPECTED_NOTICE_IDS = {"SYN-FLOOD-001", "SYN-WATER-002"}
MANIFEST_TARGETS = {
    ".gitignore",
    "DEPENDENCIES.md",
    "SECURITY_AND_PRIVACY.md",
    "assets/cover.png",
    "build_demo.py",
    "dataset-metadata.example.json",
    "dependencies/LICENSE.keras-hub.txt",
    "dependencies/keras_hub-0.28.0.audit.json",
    "LICENSE",
    "README.md",
    "fixtures/synthetic_flood_notice.json",
    "fixtures/synthetic_water_notice.json",
    KERNEL_METADATA_NAME,
    NOTEBOOK_NAME,
    "requirements-kaggle.txt",
    "runtime_smoke/README.md",
    "runtime_smoke/gemma4_smoke.py",
    "runtime_smoke/kernel-metadata.example.json",
    "runtime_smoke/tests/test_gemma4_smoke.py",
    "runtime_smoke/validate_smoke.py",
    "schemas/demo_results.schema.json",
    "tests/test_validate_project.py",
    "tests/test_build_demo.py",
    "tools/audit_wheel.py",
    "validate_project.py",
}
BINARY_PUBLICATION_TARGETS = {"assets/cover.png"}

BANNED_IMPORT_ROOTS = {
    "aiohttp",
    "boto3",
    "ftplib",
    "http",
    "huggingface_hub",
    "kagglehub",
    "requests",
    "socket",
    "subprocess",
    "torch",
    "transformers",
    "urllib",
}
BANNED_CALL_NAMES = {"__import__", "compile", "eval", "exec", "open"}
BANNED_ATTRIBUTE_CALLS = {
    "call",
    "check_call",
    "check_output",
    "popen",
    "post",
    "request",
    "run",
    "system",
    "urlopen",
}
BANNED_TEXT_PATTERNS = (
    r"https?://",
    r"\b(?:curl|wget)\b",
    r"(?:^|\s)!\s*(?:pip|apt|bash|sh)\b",
    r"\bpip\s+install\b",
    r"\bos\.system\b",
    r"\bsubprocess\b",
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA256_INLINE_RE = re.compile(r"\b[0-9a-f]{64}\b", re.IGNORECASE)
ISO_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)?\b"
)
SECRET_VALUE_RE = re.compile(
    r"\b(?:ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|"
    r"AIza[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})\b"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_publication_text(relative: str, text: str) -> list[str]:
    """Reject concrete PII, secret values, and host-specific user paths."""
    errors: list[str] = []
    scrubbed = ISO_DATE_RE.sub("", SHA256_INLINE_RE.sub("", text))
    scrubbed = scrubbed.replace("0123456789abcdef", "")
    local_path_markers = ("/" + "Users" + "/", "C:" + "\\" + "Users" + "\\")
    if any(marker in text for marker in local_path_markers):
        errors.append(f"publication source contains a host-specific user path: {relative}")
    if EMAIL_RE.search(scrubbed):
        errors.append(f"publication source contains an email address: {relative}")
    if PHONE_RE.search(scrubbed):
        errors.append(f"publication source contains a phone-like value: {relative}")
    if SECRET_VALUE_RE.search(text):
        errors.append(f"publication source contains a credential-like value: {relative}")
    return errors


def validate_release_inventory() -> list[str]:
    """Require public packaging to use the exact reviewed source allowlist."""
    errors: list[str] = []
    allowed = MANIFEST_TARGETS | {"MANIFEST.sha256"}
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        relative_path = path.relative_to(ROOT)
        relative = relative_path.as_posix()
        if "__pycache__" in relative_path.parts and path.suffix == ".pyc":
            continue
        if relative not in allowed:
            errors.append(f"unexpected file outside the public release allowlist: {relative}")
    return sorted(set(errors))


def validate_publication_sources() -> list[str]:
    """Scan every reviewed public source, including the manifest itself."""
    errors: list[str] = []
    for relative in sorted(MANIFEST_TARGETS | {"MANIFEST.sha256"}):
        if relative in BINARY_PUBLICATION_TARGETS:
            continue
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        errors.extend(scan_publication_text(relative, text))
    return sorted(set(errors))


def validate_binary_assets() -> list[str]:
    """Validate the exact public PNG without decoding arbitrary data."""
    errors: list[str] = []
    for relative in sorted(BINARY_PUBLICATION_TARGETS):
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"binary asset is missing, non-file, or symlink: {relative}")
            continue
        if path.stat().st_size > 4_000_000:
            errors.append(f"binary asset exceeds size limit: {relative}")
            continue
        with path.open("rb") as handle:
            header = handle.read(24)
        if (
            len(header) != 24
            or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"
        ):
            errors.append(f"binary asset is not a canonical PNG: {relative}")
            continue
        width, height = struct.unpack(">II", header[16:24])
        if (width, height) != (1120, 560):
            errors.append(f"binary asset dimensions mismatch: {relative}")
    return errors


def scan_code_security(source: str) -> list[str]:
    """Return deterministic security findings for Python source."""
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"notebook Python syntax error: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    errors.append(f"banned import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in BANNED_IMPORT_ROOTS:
                errors.append(f"banned import: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                errors.append(f"banned call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_ATTRIBUTE_CALLS:
                errors.append(f"banned attribute call: {node.func.attr}")
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in BANNED_IMPORT_ROOTS
            ):
                errors.append(
                    f"banned attribute call: {node.func.value.id}.{node.func.attr}"
                )

    for pattern in BANNED_TEXT_PATTERNS:
        if re.search(pattern, source, flags=re.IGNORECASE | re.MULTILINE):
            errors.append(f"banned text pattern: {pattern}")
    return sorted(set(errors))


def validate_metadata(metadata: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(metadata, dict):
        return ["kernel metadata must be an object"]
    expected = {
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [WHEEL_DATASET_REF],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [MODEL_REF],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"kernel metadata {key!r} must equal {value!r}")
    kernel_id = metadata.get("id")
    if kernel_id != "YOUR_KAGGLE_USERNAME/sahaaya-cards-gemma4-offline-civic-copilot":
        errors.append("kernel metadata template owner placeholder mismatch")
    return errors


def validate_fixture(fixture: Any, path: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(fixture, dict):
        return [f"{path.name}: fixture must be an object"]
    required = {
        "schema_version",
        "notice_id",
        "synthetic",
        "issuer",
        "issued_at",
        "title",
        "body",
        "source_language",
    }
    missing = sorted(required - fixture.keys())
    if missing:
        errors.append(f"{path.name}: missing fields {missing}")
    if fixture.get("synthetic") is not True:
        errors.append(f"{path.name}: synthetic must be true")
    if fixture.get("source_language") != "en":
        errors.append(f"{path.name}: source_language must be en for this demo")
    text = json.dumps(fixture, ensure_ascii=False)
    pii_scan_text = re.sub(
        r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)?\b",
        "",
        text,
    )
    if EMAIL_RE.search(pii_scan_text):
        errors.append(f"{path.name}: email-like personal data is forbidden")
    if PHONE_RE.search(pii_scan_text):
        errors.append(f"{path.name}: phone-like personal data is forbidden")
    if re.search(r"https?://", text, flags=re.IGNORECASE):
        errors.append(f"{path.name}: URLs are forbidden")
    return errors


def validate_notebook(notebook: Any) -> tuple[list[str], str]:
    errors: list[str] = []
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        return ["notebook must be nbformat 4"], ""
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("notebook metadata must be an object")
    elif set(metadata) != {"kernelspec", "language_info"}:
        errors.append("public notebook must exclude host-specific Kaggle metadata")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        return ["notebook must contain cells"], ""

    sources: list[str] = []
    markdown_count = 0
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            markdown_count += 1
            if index != 0:
                errors.append("the single review-note markdown cell must be first")
            continue
        if cell.get("cell_type") != "code":
            errors.append(f"cell {index}: unsupported cell type")
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"cell {index}: execution_count must be null")
        if cell.get("outputs") != []:
            errors.append(f"cell {index}: outputs must be empty")
        source = cell.get("source", [])
        if isinstance(source, list):
            sources.append("".join(source))
        elif isinstance(source, str):
            sources.append(source)
        else:
            errors.append(f"cell {index}: source must be text or text lines")

    if markdown_count != 1:
        errors.append("public notebook must contain exactly one leading review-note markdown cell")
    all_source = "\n".join(sources)
    errors.extend(scan_code_security(all_source))

    required_snippets = (
        "keras_hub.models.Gemma4CausalLM.from_preset(MODEL_PATH, dtype=\"float16\")",
        "configure_verified_runtime()",
        WHEEL_SHA256,
        WHEEL_DATASET_REF,
        "_wheel_record",
        "record_entries_verified",
        "record_bytes = archive.read(record_name)",
        "if relative == record_name:",
        "set(expected) | {record_name}",
        "_read_bounded_json",
        "model JSON exceeds the 2 MB bound",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "TF_FORCE_GPU_ALLOW_GROWTH",
        "jax.devices(\"gpu\")",
        "model.compile(sampler=\"greedy\")",
        "model.generate(",
        'prompt = "<|turn>user\\n" + user_text + "<turn|>\\n<|turn>model\\n"',
        '{"prompts": [prompt]}',
        "strip_prompt=True",
        "MAX_LENGTH = 2048",
        "_reject_duplicate_object_pairs",
        "duplicate JSON key",
        "write_runtime_journal",
        "runtime_journal.json",
        MODEL_REF,
        MODEL_PATH,
        "demo_results.json",
        "raw_final_answers",
        "safety_limitations",
        "SYN-FLOOD-001",
        "SYN-WATER-002",
        "len(text.split()) > 35",
        "fact_ledger[{fact_index}].value",
        "do_not_infer[{limit_index}]",
    )
    for snippet in required_snippets:
        if snippet not in all_source:
            errors.append(f"notebook missing required snippet: {snippet}")
    for token in FORBIDDEN_NOTEBOOK_TOKENS:
        if token in all_source:
            errors.append(f"notebook contains retired or forbidden model-route token: {token}")
    allowed_imports = {
        "__future__",
        "datetime",
        "hashlib",
        "json",
        "base64",
        "csv",
        "pathlib",
        "importlib",
        "io",
        "jax",
        "keras",
        "keras_hub",
        "os",
        "re",
        "shutil",
        "stat",
        "sys",
        "tempfile",
        "time",
        "typing",
        "zipfile",
    }
    tree = ast.parse(all_source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if module.split(".", 1)[0] not in allowed_imports:
                errors.append(f"notebook import is not allowlisted: {module}")
    return errors, all_source


def extract_embedded_notices(source: str) -> list[dict[str, Any]]:
    """Read the literal fixture assignment from notebook source without executing it."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "SYNTHETIC_NOTICES"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("SYNTHETIC_NOTICES must be a literal list of objects")
        return value
    raise ValueError("SYNTHETIC_NOTICES literal assignment was not found")


def validate_runtime_artifact(
    artifact: Any,
    *,
    require_pass: bool = True,
) -> list[str]:
    """Validate the machine-readable artifact without trusting model output."""
    errors: list[str] = []
    if not isinstance(artifact, dict):
        return ["demo_results.json must be an object"]
    required_top = {
        "schema_version",
        "project",
        "generated_at_utc",
        "model_ref",
        "model_path",
        "run_configuration",
        "safety_limitations",
        "notices",
    }
    missing_top = sorted(required_top - artifact.keys())
    if missing_top:
        errors.append(f"runtime artifact missing fields {missing_top}")
    if artifact.get("schema_version") != "1.0":
        errors.append("runtime artifact schema_version must be 1.0")
    if artifact.get("project") != "Sahaaya Cards":
        errors.append("runtime artifact project mismatch")
    if artifact.get("model_ref") != MODEL_REF:
        errors.append("runtime artifact model_ref mismatch")
    if artifact.get("model_path") != MODEL_PATH:
        errors.append("runtime artifact model_path mismatch")
    limitations = artifact.get("safety_limitations")
    if not isinstance(limitations, list) or len(limitations) < 4:
        errors.append("runtime artifact must record at least four safety limitations")
    config = artifact.get("run_configuration")
    if not isinstance(config, dict):
        errors.append("runtime run_configuration must be an object")
    else:
        expected_config = {
            "framework": "keras_hub",
            "backend": "jax",
            "weight_dtype": "float16",
            "keras_hub_version": "0.28.0",
            "wheel_dataset_ref": WHEEL_DATASET_REF,
            "wheel_sha256": WHEEL_SHA256,
            "max_length": 2048,
            "sampler": "greedy",
            "strip_prompt": True,
            "tensorflow_gpu_growth": True,
            "internet_enabled": False,
            "external_apis": False,
        }
        if set(config) != set(expected_config):
            errors.append("runtime run_configuration field inventory mismatch")
        for key, value in expected_config.items():
            if config.get(key) != value:
                errors.append(f"runtime run_configuration {key!r} mismatch")

    notices = artifact.get("notices")
    if not isinstance(notices, list):
        return errors + ["runtime notices must be a list"]
    observed_ids: set[str] = set()
    for index, notice in enumerate(notices):
        prefix = f"runtime notices[{index}]"
        if not isinstance(notice, dict):
            errors.append(f"{prefix} must be an object")
            continue
        notice_id = notice.get("notice_id")
        if isinstance(notice_id, str):
            observed_ids.add(notice_id)
        else:
            errors.append(f"{prefix}.notice_id must be text")
        if not SHA256_RE.fullmatch(str(notice.get("source_sha256", ""))):
            errors.append(f"{prefix}.source_sha256 must be lowercase SHA-256")
        prompts = notice.get("prompts")
        if not isinstance(prompts, dict) or not all(
            isinstance(prompts.get(key), str) and prompts[key]
            for key in ("generator", "verifier")
        ):
            errors.append(f"{prefix}.prompts must contain generator and verifier text")
        finals = notice.get("raw_final_answers")
        if not isinstance(finals, dict) or not all(
            isinstance(finals.get(key), str) and finals[key]
            for key in ("generator", "verifier")
        ):
            errors.append(f"{prefix}.raw_final_answers must contain final generator and verifier text")
        parsed = notice.get("parsed")
        if not isinstance(parsed, dict) or not all(
            isinstance(parsed.get(key), dict) for key in ("generator", "verifier")
        ):
            errors.append(f"{prefix}.parsed must contain generator and verifier objects")
        timing = notice.get("timing_seconds")
        if not isinstance(timing, dict) or not all(
            isinstance(timing.get(key), (int, float)) and timing[key] >= 0
            for key in ("generator", "verifier", "total")
        ):
            errors.append(f"{prefix}.timing_seconds must contain non-negative timings")
        validation = notice.get("validation")
        if not isinstance(validation, dict):
            errors.append(f"{prefix}.validation must be an object")
        else:
            if not isinstance(validation.get("passed"), bool):
                errors.append(f"{prefix}.validation.passed must be boolean")
            validation_errors = validation.get("errors")
            if not isinstance(validation_errors, list) or not all(
                isinstance(item, str) for item in validation_errors
            ):
                errors.append(f"{prefix}.validation.errors must be a string list")
            if require_pass and validation.get("passed") is not True:
                errors.append(f"{prefix} did not pass the deterministic runtime gate")
            if require_pass and validation_errors:
                errors.append(f"{prefix} contains deterministic runtime errors")

    if observed_ids != EXPECTED_NOTICE_IDS:
        errors.append(
            f"runtime notice IDs must equal {sorted(EXPECTED_NOTICE_IDS)}, got {sorted(observed_ids)}"
        )
    return errors


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
            errors.append(f"MANIFEST.sha256 line {line_number} is malformed")
            continue
        relative = parts[1].lstrip("*")
        if relative.startswith("/") or ".." in Path(relative).parts:
            errors.append(f"MANIFEST.sha256 line {line_number} has unsafe path")
            continue
        if relative in entries:
            errors.append(f"MANIFEST.sha256 duplicates {relative}")
            continue
        entries[relative] = parts[0]
    if set(entries) != MANIFEST_TARGETS:
        errors.append(
            f"MANIFEST.sha256 targets must equal {sorted(MANIFEST_TARGETS)}, got {sorted(entries)}"
        )
    for relative, expected_hash in entries.items():
        target = ROOT / relative
        if not target.is_file() or target.is_symlink():
            errors.append(f"manifest target is missing, non-file, or symlink: {relative}")
        elif sha256_file(target) != expected_hash:
            errors.append(f"manifest hash mismatch: {relative}")
    return errors


def validate_renderer(path: Path) -> list[str]:
    errors: list[str] = []
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"renderer Python syntax error: {exc}"]
    allowed_imports = {
        "argparse",
        "hashlib",
        "html",
        "json",
        "pathlib",
        "re",
        "sys",
        "tempfile",
        "typing",
        "validate_project",
        "__future__",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in allowed_imports:
                    errors.append(f"renderer import is not standard-library allowlisted: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".", 1)[0] not in allowed_imports:
                errors.append(f"renderer import is not standard-library allowlisted: {module}")
    errors.extend(scan_code_security(source))
    required = (
        "validate_runtime_artifact(artifact, require_pass=True)",
        "object_pairs_hook=_reject_duplicate_pairs",
        "html.escape(value, quote=True)",
        "default-src 'none'",
        "temporary_path.replace(output)",
        "demo_results.json",
        "demo\" / \"index.html",
        "card action exceeds 35 whitespace-delimited words",
        "fact_ledger[{index}].value",
        "do_not_infer[{limit_index}]",
    )
    for snippet in required:
        if snippet not in source:
            errors.append(f"renderer missing required safety construct: {snippet}")
    return sorted(set(errors))


def validate_runtime_semantics_path(path: Path) -> list[str]:
    """Run the renderer's strict, duplicate-key-safe evidence normalization."""
    try:
        import build_demo

        build_demo._load_and_normalize(path)
    except (OSError, KeyError, TypeError, ValueError):
        return ["runtime artifact failed deep evidence validation"]
    return []


def run_static_checks(runtime_path: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    required_files = tuple(ROOT / relative for relative in sorted(MANIFEST_TARGETS)) + (
        ROOT / "MANIFEST.sha256",
    )
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    if missing:
        errors.append(f"missing required files: {missing}")
    checks["required_files"] = "PASS" if not missing else "FAIL"
    symlinks = [str(path.relative_to(ROOT)) for path in required_files if path.is_symlink()]
    if symlinks:
        errors.append(f"required files must not be symlinks: {symlinks}")
    checks["no_required_symlinks"] = "PASS" if not symlinks else "FAIL"

    metadata_path = ROOT / KERNEL_METADATA_NAME
    if metadata_path.is_file():
        metadata_errors = validate_metadata(load_json(metadata_path))
        errors.extend(metadata_errors)
        checks["kernel_metadata"] = "PASS" if not metadata_errors else "FAIL"

    notebook_path = ROOT / NOTEBOOK_NAME
    if notebook_path.is_file():
        notebook_errors, notebook_source = validate_notebook(load_json(notebook_path))
        errors.extend(notebook_errors)
        checks["notebook_static_security"] = "PASS" if not notebook_errors else "FAIL"
        checks["notebook_source_sha256"] = hashlib.sha256(
            notebook_source.encode("utf-8")
        ).hexdigest()

    fixture_ids: set[str] = set()
    fixtures: list[dict[str, Any]] = []
    fixture_error_count = 0
    for fixture_path in FIXTURE_PATHS:
        if fixture_path.is_file():
            fixture = load_json(fixture_path)
            fixtures.append(fixture)
            fixture_ids.add(str(fixture.get("notice_id", "")))
            fixture_errors = validate_fixture(fixture, fixture_path)
            fixture_error_count += len(fixture_errors)
            errors.extend(fixture_errors)
    if fixture_ids != EXPECTED_NOTICE_IDS:
        errors.append(
            f"fixture notice IDs must equal {sorted(EXPECTED_NOTICE_IDS)}, got {sorted(fixture_ids)}"
        )
        fixture_error_count += 1
    checks["synthetic_fixtures"] = "PASS" if fixture_error_count == 0 else "FAIL"
    if notebook_path.is_file() and fixtures:
        try:
            embedded = extract_embedded_notices(notebook_source)
            fixture_map = {item["notice_id"]: item for item in fixtures}
            embedded_map = {item["notice_id"]: item for item in embedded}
            if embedded_map != fixture_map:
                errors.append("embedded notebook notices must exactly match the fixture files")
                checks["embedded_fixture_parity"] = "FAIL"
            else:
                checks["embedded_fixture_parity"] = "PASS"
        except (KeyError, TypeError, ValueError, SyntaxError) as exc:
            errors.append(f"could not validate embedded fixture parity: {exc}")
            checks["embedded_fixture_parity"] = "FAIL"

    license_path = ROOT / "LICENSE"
    if license_path.is_file():
        license_text = license_path.read_text(encoding="utf-8")
        if "Apache License" not in license_text or "Version 2.0" not in license_text:
            errors.append("LICENSE must contain Apache License 2.0")
        checks["license"] = "PASS" if "Apache License" in license_text else "FAIL"

    manifest_path = ROOT / "MANIFEST.sha256"
    if manifest_path.is_file():
        manifest_errors = validate_manifest(manifest_path)
        errors.extend(manifest_errors)
        checks["integrity_manifest"] = "PASS" if not manifest_errors else "FAIL"

    release_inventory_errors = validate_release_inventory()
    errors.extend(release_inventory_errors)
    checks["release_allowlist"] = "PASS" if not release_inventory_errors else "FAIL"

    publication_scan_errors = validate_publication_sources()
    errors.extend(publication_scan_errors)
    checks["publication_source_scan"] = "PASS" if not publication_scan_errors else "FAIL"

    binary_asset_errors = validate_binary_assets()
    errors.extend(binary_asset_errors)
    checks["binary_assets"] = "PASS" if not binary_asset_errors else "FAIL"

    renderer_path = ROOT / "build_demo.py"
    if renderer_path.is_file():
        renderer_errors = validate_renderer(renderer_path)
        errors.extend(renderer_errors)
        checks["offline_renderer_security"] = "PASS" if not renderer_errors else "FAIL"

    runtime_path = runtime_path or (ROOT / "demo_results.json")
    if runtime_path.exists():
        runtime_errors: list[str] = []
        if not runtime_path.is_file() or runtime_path.is_symlink():
            runtime_errors.append("runtime artifact must be a regular non-symlink file")
        else:
            try:
                runtime_artifact = load_json(runtime_path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                runtime_errors.append("runtime artifact must be valid UTF-8 JSON")
            else:
                runtime_errors.extend(
                    validate_runtime_artifact(runtime_artifact, require_pass=True)
                )
            semantic_errors = validate_runtime_semantics_path(runtime_path)
            runtime_errors.extend(semantic_errors)
            checks["runtime_semantic_evidence"] = (
                "PASS" if not semantic_errors else "FAIL"
            )
            if not runtime_errors:
                checks["runtime_artifact_sha256"] = sha256_file(runtime_path)
        errors.extend(runtime_errors)
        checks["runtime_artifact"] = "PASS" if not runtime_errors else "FAIL"
    else:
        warnings.append("demo_results.json is absent until the authorized Kaggle GPU run")
        checks["runtime_artifact"] = "NOT_RUN"

    return {
        "static_status": "GO" if not errors else "HOLD",
        "publication_status": "GO" if not errors and runtime_path.exists() else "HOLD",
        "checks": checks,
        "errors": sorted(set(errors)),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation for Sahaaya Cards sources and runtime evidence"
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        help="validate a downloaded demo_results.json without copying it into the source tree",
    )
    args = parser.parse_args(argv)
    report = run_static_checks(args.runtime)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["static_status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
