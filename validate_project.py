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
OWNER_PLACEHOLDER = "YOUR_KAGGLE_USERNAME"
WHEEL_DATASET_SLUG = "verified-keras-hub-028-gemma4"
WHEEL_DATASET_REF = f"{OWNER_PLACEHOLDER}/{WHEEL_DATASET_SLUG}"
KERNEL_METADATA_NAME = "kernel-metadata.example.json"
WHEEL_SHA256 = "a28cb601f7fffb7f28add1bae8110459fc3ac7d9e2159453dfbda9e97271fc87"
FORBIDDEN_NOTEBOOK_TOKENS = (
    "Gemma3CausalLM",
    "gemma3_instruct",
    "AutoProcessor",
    "AutoModelForCausalLM",
    "AutoModelForMultimodalLM",
    "apply_chat_template",
    "KERAS_BACKEND\": \"jax",
    "jax.devices",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
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
    "ILLUSTRATIVE_OUTPUT.md",
    "SECURITY_AND_PRIVACY.md",
    "assets/cover.png",
    "build_demo.py",
    "dataset-metadata.example.json",
    "dependencies/LICENSE.keras-hub.txt",
    "dependencies/keras_hub-0.28.0.audit.json",
    "LICENSE",
    "README.md",
    "WRITEUP_DRAFT.md",
    "fixtures/synthetic_flood_notice.json",
    "fixtures/synthetic_water_notice.json",
    KERNEL_METADATA_NAME,
    NOTEBOOK_NAME,
    "requirements-kaggle.txt",
    "schemas/demo_results.schema.json",
    "tests/test_validate_project.py",
    "tests/test_build_demo.py",
    "tests/test_audit_wheel.py",
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
    "jax",
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
RUNTIME_OWNER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,38}$")


def _wheel_dataset_owner(value: Any, *, allow_placeholder: bool) -> str | None:
    """Return a safe owner only for the one pinned wheel-Dataset slug."""
    if not isinstance(value, str) or value.count("/") != 1:
        return None
    owner, slug = value.split("/", 1)
    if slug != WHEEL_DATASET_SLUG:
        return None
    if owner == OWNER_PLACEHOLDER:
        return owner if allow_placeholder else None
    return owner if RUNTIME_OWNER_RE.fullmatch(owner) else None


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
        # Repository bookkeeping is outside the release payload.  Ignoring it
        # lets the same validator run both in a clean export and in a clone.
        if relative_path.parts and relative_path.parts[0] == ".git":
            continue
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
    """Validate the bounded public cover without decoding arbitrary data."""
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
        "title": "Sahaaya Cards Gemma 4 Offline Civic Copilot",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "nvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [MODEL_REF],
    }
    if set(metadata) != set(expected) | {"id", "dataset_sources"}:
        errors.append("kernel metadata field inventory mismatch")
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"kernel metadata {key!r} must equal {value!r}")
    dataset_sources = metadata.get("dataset_sources")
    dataset_ref = (
        dataset_sources[0]
        if isinstance(dataset_sources, list) and len(dataset_sources) == 1
        else None
    )
    dataset_owner = _wheel_dataset_owner(dataset_ref, allow_placeholder=True)
    if dataset_owner is None:
        errors.append("kernel metadata wheel Dataset reference is invalid")
    kernel_id = metadata.get("id")
    if not isinstance(kernel_id, str) or kernel_id.count("/") != 1:
        errors.append("kernel metadata id is invalid")
    else:
        kernel_owner, kernel_slug = kernel_id.split("/", 1)
        if (
            kernel_slug != "sahaaya-cards-gemma4-offline-civic-copilot"
            or _wheel_dataset_owner(
                f"{kernel_owner}/{WHEEL_DATASET_SLUG}",
                allow_placeholder=True,
            )
            is None
        ):
            errors.append("kernel metadata id is invalid")
        elif dataset_owner is not None and kernel_owner != dataset_owner:
            errors.append("kernel and wheel Dataset owners must match")
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
            markdown = "".join(cell.get("source", []))
            required_status = (
                "NOT RUNTIME-PROVEN",
                "V4",
                "V5",
                "V6",
                "not model-generated",
            )
            for phrase in required_status:
                if phrase not in markdown:
                    errors.append(f"public notebook status note missing: {phrase}")
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
    wheel_ref: str | None = None
    try:
        notebook_tree = ast.parse(all_source)
    except SyntaxError:
        notebook_tree = None
    if notebook_tree is not None:
        for node in notebook_tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "WHEEL_DATASET_REF"
                for target in node.targets
            ):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                wheel_ref = node.value.value
            break
    if _wheel_dataset_owner(wheel_ref, allow_placeholder=True) is None:
        errors.append("notebook wheel Dataset reference is invalid")
    elif f"/kaggle/input/datasets/{wheel_ref}" not in all_source:
        errors.append("notebook wheel Dataset path does not match its reference")

    required_snippets = (
        "keras_hub.models.Gemma4CausalLM.from_preset(",
        "configure_verified_runtime()",
        WHEEL_SHA256,
        "_wheel_record",
        "ALLOWED_WHEEL_FILE_SUFFIXES",
        "record_entries_verified",
        "record_bytes = archive.read(record_name)",
        "if relative == record_name:",
        "set(expected) | {record_name}",
        "_read_bounded_json",
        "model JSON exceeds the 2 MB bound",
        "TF_FORCE_GPU_ALLOW_GROWTH",
        '"KERAS_BACKEND": "torch"',
        'tf.config.set_visible_devices([], "GPU")',
        'tf.config.get_visible_devices("GPU")',
        "torch.cuda.device_count() != 2",
        '"T4" not in name.upper()',
        'keras.config.set_dtype_policy("float16")',
        'with keras.device("gpu:0")',
        'dtype="float16"',
        "load_weights=True",
        "_variable_inventory",
        'by_device != {"cuda:0": total}',
        "model.compile(sampler=\"greedy\")",
        "model.generate(",
        'prompt = "<|turn>user\\n" + user_text + "<turn|>\\n<|turn>model\\n"',
        '{"prompts": [prompt]}',
        "strip_prompt=True",
        "GENERATOR_MAX_LENGTH = 2048",
        "VERIFIER_MAX_LENGTH = 3072",
        "MIN_COMPLETION_TOKENS = 512",
        "prompt_budget(",
        "tokenizer prompt budget leaves fewer than 512 completion tokens",
        "_reject_duplicate_object_pairs",
        "duplicate JSON key",
        "write_runtime_journal",
        "runtime_journal.json",
        'artifact["status"] = "PASS"',
        'len(artifact["notices"]) == 2',
        "artifact_sha256 = sha256_file(OUTPUT_PATH)",
        'write_runtime_journal(\n        final_stage,',
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
        "keras",
        "keras_hub",
        "os",
        "re",
        "shutil",
        "stat",
        "sys",
        "tempfile",
        "tensorflow",
        "time",
        "torch",
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
        "status",
        "model_ref",
        "model_path",
        "run_configuration",
        "runtime_provenance",
        "safety_limitations",
        "failures",
        "notices",
    }
    if set(artifact) != required_top:
        errors.append("runtime artifact field inventory mismatch")
    if artifact.get("schema_version") != "1.0":
        errors.append("runtime artifact schema_version must be 1.0")
    if artifact.get("project") != "Sahaaya Cards":
        errors.append("runtime artifact project mismatch")
    status = artifact.get("status")
    if status not in {"PARTIAL", "FAIL", "PASS"}:
        errors.append("runtime artifact status is invalid")
    if require_pass and status != "PASS":
        errors.append("runtime artifact status is not PASS")
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
            "backend": "torch",
            "weight_dtype": "float16",
            "keras_hub_version": "0.28.0",
            "wheel_sha256": WHEEL_SHA256,
            "generator_max_length": 2048,
            "verifier_max_length": 3072,
            "minimum_completion_tokens": 512,
            "sampler": "greedy",
            "strip_prompt": True,
            "gpu": "T4x2",
            "tensorflow_gpu_visible": False,
            "internet_enabled": False,
            "external_apis": False,
        }
        if set(config) != set(expected_config) | {"wheel_dataset_ref"}:
            errors.append("runtime run_configuration field inventory mismatch")
        for key, value in expected_config.items():
            if config.get(key) != value:
                errors.append(f"runtime run_configuration {key!r} mismatch")
        wheel_owner = _wheel_dataset_owner(
            config.get("wheel_dataset_ref"),
            allow_placeholder=not require_pass,
        )
        if wheel_owner is None:
            errors.append("runtime wheel Dataset reference is invalid")

    failures = artifact.get("failures")
    if not isinstance(failures, list):
        errors.append("runtime failures must be a list")
    else:
        for index, failure in enumerate(failures):
            if not isinstance(failure, dict) or set(failure) != {
                "error_type", "failure_classification", "stage"
            }:
                errors.append(f"runtime failures[{index}] field inventory mismatch")
                continue
            if any(
                not isinstance(failure.get(key), str)
                or not failure[key]
                or len(failure[key]) > 80
                for key in ("error_type", "failure_classification", "stage")
            ):
                errors.append(f"runtime failures[{index}] is not bounded text")
        if require_pass and failures:
            errors.append("PASS runtime artifact must contain no failures")

    provenance = artifact.get("runtime_provenance")
    if require_pass:
        expected_provenance_keys = {
            "backend", "explicit_model_dtype", "global_policy", "gpu",
            "gpu_count", "keras_hub_version", "load_seconds",
            "memory_after_load", "memory_before_load", "model",
            "model_variable_bytes", "official_weights_loaded",
            "tensorflow_gpu_visible", "torch_version",
            "variable_bytes_by_device", "variable_bytes_by_dtype", "weight_dtype",
            "wheel",
        }
        if not isinstance(provenance, dict) or set(provenance) != expected_provenance_keys:
            errors.append("runtime provenance field inventory mismatch")
        else:
            fixed = {
                "backend": "torch",
                "explicit_model_dtype": "float16",
                "global_policy": "float16",
                "gpu": "T4x2",
                "gpu_count": 2,
                "keras_hub_version": "0.28.0",
                "official_weights_loaded": True,
                "tensorflow_gpu_visible": False,
                "weight_dtype": "float16",
            }
            if any(provenance.get(key) != value for key, value in fixed.items()):
                errors.append("runtime provenance contract mismatch")
            torch_version = provenance.get("torch_version")
            if not isinstance(torch_version, str) or not torch_version or len(torch_version) > 64:
                errors.append("runtime Torch version is invalid")
            load_seconds = provenance.get("load_seconds")
            if not isinstance(load_seconds, (int, float)) or isinstance(load_seconds, bool) or not 0 < load_seconds <= 1800:
                errors.append("runtime model load duration is invalid")
            wheel = provenance.get("wheel")
            expected_wheel = {
                "archive_bytes": 1_525_623,
                "member_count": 766,
                "record_entries_verified": 765,
                "sha256": WHEEL_SHA256,
            }
            if wheel != expected_wheel:
                errors.append("runtime audited-wheel provenance mismatch")
            model = provenance.get("model")
            if not isinstance(model, dict) or set(model) != {
                "file_count", "inventory_sha256", "total_bytes"
            }:
                errors.append("runtime model provenance field inventory mismatch")
            else:
                if not isinstance(model.get("file_count"), int) or isinstance(model.get("file_count"), bool) or not 6 <= model["file_count"] <= 32:
                    errors.append("runtime model file count is invalid")
                if not isinstance(model.get("total_bytes"), int) or isinstance(model.get("total_bytes"), bool) or not 8_000_000_000 <= model["total_bytes"] <= 13_000_000_000:
                    errors.append("runtime model byte count is invalid")
                if not SHA256_RE.fullmatch(str(model.get("inventory_sha256", ""))):
                    errors.append("runtime model inventory SHA-256 is invalid")
            by_dtype = provenance.get("variable_bytes_by_dtype")
            if not isinstance(by_dtype, dict) or not by_dtype:
                errors.append("runtime variable dtype inventory is missing")
            elif any(
                not isinstance(key, str)
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for key, value in by_dtype.items()
            ):
                errors.append("runtime variable dtype inventory is invalid")
            else:
                if by_dtype.get("float16", 0) < 8_000_000_000:
                    errors.append("runtime float16 variable inventory is too small")
                if any(value > 0 for key, value in by_dtype.items() if key.startswith("float") and key != "float16"):
                    errors.append("runtime variable inventory contains non-float16 floating values")
                variable_bytes = provenance.get("model_variable_bytes")
                if variable_bytes != sum(by_dtype.values()) or not isinstance(variable_bytes, int) or isinstance(variable_bytes, bool) or not 8_000_000_000 <= variable_bytes <= 13_000_000_000:
                    errors.append("runtime model variable byte binding is invalid")
                if provenance.get("variable_bytes_by_device") != {"cuda:0": variable_bytes}:
                    errors.append("runtime model variables are not bound to CUDA device 0")
            memory_keys = {
                "allocated_bytes", "free_bytes", "peak_allocated_bytes",
                "reserved_bytes", "total_bytes",
            }
            for key in ("memory_before_load", "memory_after_load"):
                memory = provenance.get(key)
                if not isinstance(memory, dict) or set(memory) != memory_keys:
                    errors.append(f"runtime {key} inventory mismatch")
                elif any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in memory.values()):
                    errors.append(f"runtime {key} contains an invalid byte count")
                elif memory["total_bytes"] < 12_000_000_000:
                    errors.append(f"runtime {key} reports an unexpectedly small GPU")
            after = provenance.get("memory_after_load")
            if isinstance(after, dict) and isinstance(after.get("free_bytes"), int) and after["free_bytes"] < 2_000_000_000:
                errors.append("runtime generation ran below the free-memory gate")
    elif not isinstance(provenance, dict):
        errors.append("runtime provenance must be an object")

    notices = artifact.get("notices")
    if not isinstance(notices, list):
        return errors + ["runtime notices must be a list"]
    observed_ids: set[str] = set()
    for index, notice in enumerate(notices):
        prefix = f"runtime notices[{index}]"
        if not isinstance(notice, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if set(notice) != {
            "notice_id",
            "source_sha256",
            "prompts",
            "raw_final_answers",
            "parsed",
            "token_budgets",
            "timing_seconds",
            "validation",
        }:
            errors.append(f"{prefix} field inventory mismatch")
        notice_id = notice.get("notice_id")
        if isinstance(notice_id, str):
            observed_ids.add(notice_id)
        else:
            errors.append(f"{prefix}.notice_id must be text")
        if not SHA256_RE.fullmatch(str(notice.get("source_sha256", ""))):
            errors.append(f"{prefix}.source_sha256 must be lowercase SHA-256")
        prompts = notice.get("prompts")
        if (
            not isinstance(prompts, dict)
            or set(prompts) != {"generator", "verifier"}
            or not all(
            isinstance(prompts.get(key), str) and prompts[key]
            for key in ("generator", "verifier")
            )
        ):
            errors.append(f"{prefix}.prompts must contain generator and verifier text")
        finals = notice.get("raw_final_answers")
        if (
            not isinstance(finals, dict)
            or set(finals) != {"generator", "verifier"}
            or not all(
            isinstance(finals.get(key), str) and finals[key]
            for key in ("generator", "verifier")
            )
        ):
            errors.append(f"{prefix}.raw_final_answers must contain final generator and verifier text")
        parsed = notice.get("parsed")
        if (
            not isinstance(parsed, dict)
            or set(parsed) != {"generator", "verifier"}
            or not all(
            isinstance(parsed.get(key), dict) for key in ("generator", "verifier")
            )
        ):
            errors.append(f"{prefix}.parsed must contain generator and verifier objects")
        budgets = notice.get("token_budgets")
        if not isinstance(budgets, dict) or set(budgets) != {"generator", "verifier"}:
            errors.append(f"{prefix}.token_budgets field inventory mismatch")
        elif require_pass:
            for role, expected_max in (("generator", 2048), ("verifier", 3072)):
                budget = budgets.get(role)
                if not isinstance(budget, dict) or set(budget) != {
                    "completion_budget_tokens", "max_length",
                    "minimum_completion_tokens", "prompt_tokens",
                }:
                    errors.append(f"{prefix}.token_budgets.{role} field inventory mismatch")
                    continue
                values = tuple(budget.get(key) for key in (
                    "completion_budget_tokens", "max_length",
                    "minimum_completion_tokens", "prompt_tokens",
                ))
                if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
                    errors.append(f"{prefix}.token_budgets.{role} must contain integers")
                elif (
                    budget["max_length"] != expected_max
                    or budget["minimum_completion_tokens"] != 512
                    or budget["prompt_tokens"] <= 0
                    or budget["completion_budget_tokens"] < 512
                    or budget["prompt_tokens"] + budget["completion_budget_tokens"] != expected_max
                ):
                    errors.append(f"{prefix}.token_budgets.{role} violates the completion reserve")
        timing = notice.get("timing_seconds")
        if (
            not isinstance(timing, dict)
            or set(timing) != {"generator", "verifier", "total"}
            or not all(
                isinstance(timing.get(key), (int, float))
                and not isinstance(timing.get(key), bool)
                and timing[key] >= 0
                for key in ("generator", "verifier", "total")
            )
        ):
            errors.append(f"{prefix}.timing_seconds must contain non-negative timings")
        validation = notice.get("validation")
        if not isinstance(validation, dict) or set(validation) != {"passed", "errors"}:
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

    if require_pass and (observed_ids != EXPECTED_NOTICE_IDS or len(notices) != 2):
        errors.append(
            f"runtime notice IDs must equal {sorted(EXPECTED_NOTICE_IDS)}, got {sorted(observed_ids)}"
        )
    elif not require_pass and (
        not observed_ids.issubset(EXPECTED_NOTICE_IDS)
        or len(observed_ids) != len(notices)
    ):
        errors.append("diagnostic runtime notice IDs must be unique expected synthetic IDs")
    return errors


def validate_runtime_journal(journal: Any, artifact_sha256: str) -> list[str]:
    """Require a final journal that binds the exact publication artifact."""
    errors: list[str] = []
    if not isinstance(journal, dict) or set(journal) != {
        "schema_version", "stage", "updated_at_utc", "details"
    }:
        return ["runtime journal field inventory mismatch"]
    if journal.get("schema_version") != "1.0" or journal.get("stage") != "complete":
        errors.append("runtime journal did not complete")
    if not isinstance(journal.get("updated_at_utc"), str) or not journal["updated_at_utc"]:
        errors.append("runtime journal timestamp is missing")
    details = journal.get("details")
    expected = {
        "artifact_sha256": artifact_sha256,
        "completed_notices": 2,
        "error_count": 0,
        "status": "PASS",
    }
    if details != expected:
        errors.append("runtime journal does not bind the final PASS artifact")
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
        "parse_constant=_reject_non_finite",
        "_exact_object(",
        "_parse_raw_object(",
        "raw and parsed {role} answers differ",
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


def run_static_checks(
    runtime_path: Path | None = None,
    journal_path: Path | None = None,
) -> dict[str, Any]:
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

    writeup_path = ROOT / "WRITEUP_DRAFT.md"
    if writeup_path.is_file():
        writeup_words = re.findall(r"\b[\w’'-]+\b", writeup_path.read_text(encoding="utf-8"))
        checks["writeup_word_count"] = len(writeup_words)
        if len(writeup_words) >= 1500:
            errors.append(f"WRITEUP_DRAFT.md must remain below 1500 words, got {len(writeup_words)}")

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
        resolved_journal = journal_path or (
            runtime_path.parent / "runtime_journal.json"
        )
        journal_errors: list[str] = []
        if not resolved_journal.is_file() or resolved_journal.is_symlink():
            journal_errors.append(
                "runtime_journal.json must be a regular non-symlink file beside the artifact or supplied explicitly"
            )
        else:
            try:
                runtime_journal = load_json(resolved_journal)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                journal_errors.append("runtime journal must be valid UTF-8 JSON")
            else:
                journal_errors.extend(
                    validate_runtime_journal(
                        runtime_journal,
                        sha256_file(runtime_path),
                    )
                )
        errors.extend(journal_errors)
        checks["runtime_journal_binding"] = (
            "PASS" if not journal_errors else "FAIL"
        )
    else:
        if journal_path is not None:
            errors.append("--journal cannot be validated without --runtime")
        warnings.append(
            "no validated model artifact exists; V4-V6 remained incomplete under the available Kaggle T4x2 memory"
        )
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
    parser.add_argument(
        "--journal",
        type=Path,
        help="validate the downloaded runtime_journal.json that binds --runtime",
    )
    args = parser.parse_args(argv)
    report = run_static_checks(args.runtime, args.journal)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["static_status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
