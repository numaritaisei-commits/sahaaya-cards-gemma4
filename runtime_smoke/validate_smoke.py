"""Static and output validator for the fail-closed Gemma 4 smoke."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


EXPECTED_SHA = "a28cb601f7fffb7f28add1bae8110459fc3ac7d9e2159453dfbda9e97271fc87"
EXPECTED_MODEL_SOURCE = "keras/gemma4/keras/gemma4_instruct_2b/2"
EXPECTED_DATASET_SOURCE = "YOUR_KAGGLE_USERNAME/verified-keras-hub-028-gemma4"
EXPECTED_WHEEL_REPORT = {
    "archive_bytes": 1_525_623,
    "member_count": 766,
    "record_entries_verified": 765,
    "record_name": "keras_hub-0.28.0.dist-info/RECORD",
    "sha256": EXPECTED_SHA,
    "uncompressed_bytes": 5_887_722,
}
MIN_MODEL_BYTES = 8_000_000_000
MAX_MODEL_BYTES = 13_000_000_000
MAX_MODEL_FILES = 32
ALLOWED_IMPORTS = {
    "__future__", "argparse", "ast", "base64", "csv", "hashlib",
    "importlib", "io", "jax", "json", "keras", "keras_hub", "os",
    "pathlib", "shutil", "stat", "sys", "tempfile", "zipfile",
}
BANNED_CALLS = {
    "__import__", "compile", "eval", "exec", "importlib.import_module",
    "os.popen", "os.system", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.Popen", "subprocess.run",
}


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path}")


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_json(path: Path):
    _regular(path)
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def validate_source(path: Path) -> None:
    _regular(path)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {item.name.split(".", 1)[0] for item in node.names}
            if roots - ALLOWED_IMPORTS:
                raise ValueError("unapproved import")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise ValueError("unapproved from-import")
        elif isinstance(node, ast.Call) and _dotted(node.func) in BANNED_CALLS:
            raise ValueError("forbidden execution primitive")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "http://" in node.value or "https://" in node.value:
                raise ValueError("network URL embedded in smoke")
    required = [
        EXPECTED_SHA,
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "TF_FORCE_GPU_ALLOW_GROWTH",
        'dtype="float16"',
        'sampler="greedy"',
        "strip_prompt=True",
        "generated_sha256",
    ]
    if any(value not in text for value in required):
        raise ValueError("required fail-closed control is missing")


def validate_metadata(path: Path) -> None:
    data = _load_json(path)
    expected = {
        "code_file": "gemma4_smoke.py",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [EXPECTED_DATASET_SOURCE],
        "competition_sources": ["build-with-gemma-ieee-cs-srmist-ktr"],
        "kernel_sources": [],
        "model_sources": ["keras/gemma4/keras/gemma4_instruct_2b/2"],
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"unsafe metadata field: {key}")


def validate_report(path: Path) -> str:
    report = _load_json(path)
    if set(report) != {
        "internet_enabled",
        "model",
        "model_source",
        "runtime",
        "schema_version",
        "status",
        "wheel",
        "wheel_dataset_source",
    }:
        raise ValueError("smoke report field inventory mismatch")
    if report.get("schema_version") != "1.0" or report.get("status") != "PASS":
        raise ValueError("smoke did not pass")
    if report.get("internet_enabled") is not False:
        raise ValueError("internet was enabled")
    if report.get("model_source") != EXPECTED_MODEL_SOURCE:
        raise ValueError("model source mismatch")
    if report.get("wheel_dataset_source") != EXPECTED_DATASET_SOURCE:
        raise ValueError("wheel dataset source mismatch")

    wheel = report.get("wheel")
    if wheel != EXPECTED_WHEEL_REPORT:
        raise ValueError("wheel report mismatch")

    model = report.get("model")
    if not isinstance(model, dict) or set(model) != {
        "file_count",
        "inventory_sha256",
        "total_bytes",
    }:
        raise ValueError("model report field inventory mismatch")
    if not isinstance(model.get("file_count"), int) or isinstance(
        model.get("file_count"), bool
    ) or not 6 <= model["file_count"] <= MAX_MODEL_FILES:
        raise ValueError("model file count mismatch")
    if not isinstance(model.get("total_bytes"), int) or isinstance(
        model.get("total_bytes"), bool
    ) or not MIN_MODEL_BYTES <= model["total_bytes"] <= MAX_MODEL_BYTES:
        raise ValueError("model size mismatch")
    if not _is_sha256(model.get("inventory_sha256")):
        raise ValueError("model inventory digest mismatch")

    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "backend",
        "generated_codepoints",
        "generated_sha256",
        "generated_utf8_bytes",
        "gpu",
        "keras_hub_version",
        "max_length",
        "sampler",
        "tensorflow_gpu_growth",
        "weight_dtype",
    }:
        raise ValueError("runtime field inventory mismatch")
    expected = {
        "backend": "jax", "gpu": "P100", "keras_hub_version": "0.28.0",
        "max_length": 96, "sampler": "greedy", "tensorflow_gpu_growth": True,
        "weight_dtype": "float16",
    }
    if any(runtime.get(key) != value for key, value in expected.items()):
        raise ValueError("runtime contract mismatch")
    if not isinstance(runtime.get("generated_utf8_bytes"), int) or isinstance(
        runtime.get("generated_utf8_bytes"), bool
    ) or not (
        0 < runtime["generated_utf8_bytes"] <= 16_384
    ):
        raise ValueError("invalid generated length")
    if not isinstance(runtime.get("generated_codepoints"), int) or isinstance(
        runtime.get("generated_codepoints"), bool
    ) or not (
        0 < runtime["generated_codepoints"] <= runtime["generated_utf8_bytes"]
    ):
        raise ValueError("invalid generated codepoint count")
    digest = runtime.get("generated_sha256")
    if not _is_sha256(digest):
        raise ValueError("invalid generated digest")
    serialized = json.dumps(report)
    if "generated_text" in serialized or "completion" in serialized:
        raise ValueError("generated body leaked into report")
    return digest


def validate_journal(path: Path, generated_sha256: str) -> None:
    journal = _load_json(path)
    if set(journal) != {"details", "schema_version", "stage"}:
        raise ValueError("smoke journal field inventory mismatch")
    if journal.get("schema_version") != "1.0" or journal.get("stage") != "complete":
        raise ValueError("smoke journal did not complete")
    details = journal.get("details")
    if details != {"generated_sha256": generated_sha256, "status": "PASS"}:
        raise ValueError("smoke journal/report mismatch")


def validate_manifest(path: Path) -> None:
    _regular(path)
    root = path.parent
    rows = [line.split("  ", 1) for line in path.read_text().splitlines() if line]
    expected = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != path and "__pycache__" not in p.parts}
    if {name for _, name in rows} != expected:
        raise ValueError("manifest inventory mismatch")
    for digest, name in rows:
        _regular(root / name)
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("manifest hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    validate_source(args.source)
    validate_metadata(args.metadata)
    if bool(args.report) != bool(args.journal):
        parser.error("--report and --journal must be supplied together")
    if args.report and args.journal:
        generated_sha256 = validate_report(args.report)
        validate_journal(args.journal, generated_sha256)
    if args.manifest:
        validate_manifest(args.manifest)
    print("PASS: Gemma 4 smoke validation")


if __name__ == "__main__":
    main()
