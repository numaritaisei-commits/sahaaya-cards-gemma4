"""Fail-closed Gemma 4 runtime smoke for a private Kaggle P100 session.

The script uses only attached Kaggle inputs. It never prints or stores generated
text; the report contains only output lengths and a SHA-256 digest.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath


EXPECTED_WHEEL_FILENAME = "keras_hub-0.28.0-py3-none-any.whl"
EXPECTED_WHEEL_SHA256 = (
    "a28cb601f7fffb7f28add1bae8110459fc3ac7d9e2159453dfbda9e97271fc87"
)
EXPECTED_KERAS_HUB_VERSION = "0.28.0"
EXPECTED_WHEEL_TOP_LEVEL = {
    "keras_hub",
    "keras_hub-0.28.0.dist-info",
}
WHEEL_DATASET_ROOT = Path("/kaggle/input/verified-keras-hub-028-gemma4")
WHEEL_PATH = WHEEL_DATASET_ROOT / EXPECTED_WHEEL_FILENAME
MODEL_ROOT = Path(
    "/kaggle/input/models/keras/gemma4/keras/gemma4_instruct_2b/2"
)
WORKING_ROOT = Path("/kaggle/working")
VENDOR_ROOT = WORKING_ROOT / "vendor"
REPORT_PATH = WORKING_ROOT / "gemma4_smoke_report.json"
JOURNAL_PATH = WORKING_ROOT / "gemma4_smoke_journal.json"

EXPECTED_MODEL_FILES = {
    "assets/tokenizer/vocabulary.spm",
    "config.json",
    "model.weights.json",
    "model_00000.weights.h5",
    "model_00001.weights.h5",
    "task.json",
}
ALLOWED_MODEL_SUFFIXES = {".h5", ".json", ".spm"}
FORBIDDEN_WHEEL_SUFFIXES = {
    ".a",
    ".class",
    ".dylib",
    ".dll",
    ".exe",
    ".jar",
    ".node",
    ".o",
    ".pyd",
    ".pyc",
    ".so",
}
MAX_WHEEL_ARCHIVE_BYTES = 2_000_000
MAX_WHEEL_MEMBER_BYTES = 2_000_000
MAX_WHEEL_UNCOMPRESSED_BYTES = 12_000_000
MIN_MODEL_BYTES = 8_000_000_000
MAX_MODEL_BYTES = 13_000_000_000
MAX_MODEL_FILES = 32
MAX_JSON_BYTES = 2_000_000
MAX_GENERATED_UTF8_BYTES = 16_384
MAX_LENGTH = 96
PROMPT = (
    "<|turn>user\n"
    "Reply with one short word confirming readiness.<turn|>\n"
    "<|turn>model\n"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return encoded.rstrip(b"=").decode("ascii")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")


def _require_regular_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory")


def _safe_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename
    member = PurePosixPath(name)
    if not name or name.startswith("/") or member.is_absolute():
        raise ValueError("unsafe absolute or empty ZIP member path")
    if ".." in member.parts or "\\" in name or "\x00" in name:
        raise ValueError("unsafe ZIP member path")
    if not member.parts or member.parts[0] in {"", "."}:
        raise ValueError("malformed ZIP member path")

    mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise ValueError("symlink wheel member is forbidden")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError("special wheel member is forbidden")
    if not info.is_dir() and mode & 0o111:
        raise ValueError("executable wheel member is forbidden")
    if member.suffix.lower() in FORBIDDEN_WHEEL_SUFFIXES:
        raise ValueError("native or executable wheel payload is forbidden")
    if info.file_size < 0 or info.file_size > MAX_WHEEL_MEMBER_BYTES:
        raise ValueError("wheel member exceeds the size limit")
    return member


def _read_record(
    archive: zipfile.ZipFile,
    file_names: set[str],
) -> tuple[dict[str, tuple[str, int]], str]:
    record_name = "keras_hub-0.28.0.dist-info/RECORD"
    if record_name not in file_names:
        raise ValueError("wheel RECORD is missing")
    record_text = archive.read(record_name).decode("utf-8")
    rows = list(csv.reader(io.StringIO(record_text)))
    if not rows:
        raise ValueError("wheel RECORD is empty")

    seen: set[str] = set()
    expected: dict[str, tuple[str, int]] = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError("malformed RECORD row")
        name, hash_field, size_field = row
        if name in seen:
            raise ValueError("duplicate RECORD path")
        seen.add(name)
        if name == record_name:
            if hash_field or size_field:
                raise ValueError("RECORD self-entry must be unhashed")
            continue
        if not hash_field.startswith("sha256=") or not size_field.isdigit():
            raise ValueError("RECORD entry must have SHA-256 and size")
        expected[name] = (
            hash_field.removeprefix("sha256="),
            int(size_field),
        )

    if seen != file_names:
        raise ValueError("RECORD inventory mismatch")
    return expected, record_name


def audit_wheel(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_WHEEL_SHA256,
    expected_filename: str = EXPECTED_WHEEL_FILENAME,
) -> dict[str, object]:
    """Verify archive identity, paths, types, metadata, and every RECORD hash."""

    _require_regular_file(path, "wheel")
    if path.name != expected_filename:
        raise ValueError("unexpected wheel filename")
    archive_bytes = path.stat().st_size
    if archive_bytes > MAX_WHEEL_ARCHIVE_BYTES:
        raise ValueError("wheel archive exceeds the size limit")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError("wheel SHA-256 mismatch")

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos:
            raise ValueError("wheel is empty")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")

        total_size = 0
        top_levels: set[str] = set()
        file_names: set[str] = set()
        for info in infos:
            member = _safe_zip_member(info)
            total_size += info.file_size
            top_levels.add(member.parts[0])
            if not info.is_dir():
                file_names.add(info.filename)
        if total_size > MAX_WHEEL_UNCOMPRESSED_BYTES:
            raise ValueError("wheel expands beyond the size limit")
        if top_levels != EXPECTED_WHEEL_TOP_LEVEL:
            raise ValueError("unexpected wheel top-level inventory")

        metadata_name = "keras_hub-0.28.0.dist-info/METADATA"
        wheel_name = "keras_hub-0.28.0.dist-info/WHEEL"
        if metadata_name not in file_names or wheel_name not in file_names:
            raise ValueError("wheel package metadata is incomplete")
        metadata_text = archive.read(metadata_name).decode("utf-8")
        wheel_text = archive.read(wheel_name).decode("utf-8")
        if "Name: keras-hub\n" not in metadata_text:
            raise ValueError("wheel package name mismatch")
        if "Version: 0.28.0\n" not in metadata_text:
            raise ValueError("wheel package version mismatch")
        if "Requires-Python: >=3.11\n" not in metadata_text:
            raise ValueError("wheel Python requirement mismatch")
        if "Requires-Dist: keras>=3.13\n" not in metadata_text:
            raise ValueError("wheel Keras requirement mismatch")
        if "Root-Is-Purelib: true\n" not in wheel_text:
            raise ValueError("wheel is not declared pure Python")
        if "Tag: py3-none-any\n" not in wheel_text:
            raise ValueError("wheel compatibility tag mismatch")

        expected, record_name = _read_record(archive, file_names)
        for name, (expected_hash, expected_size) in expected.items():
            data = archive.read(name)
            if len(data) != expected_size:
                raise ValueError("RECORD member size mismatch")
            if _record_digest(data) != expected_hash:
                raise ValueError("RECORD member hash mismatch")

    return {
        "archive_bytes": archive_bytes,
        "member_count": len(file_names),
        "record_entries_verified": len(expected),
        "record_name": record_name,
        "sha256": digest,
        "uncompressed_bytes": total_size,
    }


def _cleanup_owned_temp(path: Path) -> None:
    expected_parent = WORKING_ROOT.resolve()
    if path.parent.resolve() != expected_parent or not path.name.startswith(
        ".gemma4-vendor-"
    ):
        raise ValueError("refusing to clean an unowned temporary directory")
    if path.exists():
        shutil.rmtree(path)


def extract_verified_wheel(
    wheel: Path,
    destination: Path,
    *,
    expected_sha256: str = EXPECTED_WHEEL_SHA256,
    expected_filename: str = EXPECTED_WHEEL_FILENAME,
) -> dict[str, object]:
    """Re-audit and atomically extract only verified regular wheel members."""

    _require_regular_directory(WORKING_ROOT, "Kaggle working root")
    if destination.exists() or destination.is_symlink():
        raise ValueError("vendor destination must not already exist")
    audit = audit_wheel(
        wheel,
        expected_sha256=expected_sha256,
        expected_filename=expected_filename,
    )
    temp_path = Path(
        tempfile.mkdtemp(prefix=".gemma4-vendor-", dir=str(WORKING_ROOT))
    )
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos = archive.infolist()
            file_names = {
                info.filename for info in infos if not info.is_dir()
            }
            expected, record_name = _read_record(archive, file_names)
            record_bytes = archive.read(record_name)
            for info in infos:
                member = _safe_zip_member(info)
                target = temp_path.joinpath(*member.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as out:
                    shutil.copyfileobj(source, out, length=1 << 20)

        extracted_files: set[str] = set()
        for base, directories, files in os.walk(temp_path, followlinks=False):
            base_path = Path(base)
            for name in directories:
                child = base_path / name
                if child.is_symlink() or not child.is_dir():
                    raise ValueError("unsafe extracted directory")
            for name in files:
                child = base_path / name
                _require_regular_file(child, "extracted wheel member")
                relative = child.relative_to(temp_path).as_posix()
                extracted_files.add(relative)
                if relative == record_name:
                    if child.read_bytes() != record_bytes:
                        raise ValueError("extracted RECORD self-entry mismatch")
                    continue
                expected_hash, expected_size = expected[relative]
                if child.stat().st_size != expected_size:
                    raise ValueError("extracted RECORD size mismatch")
                data = child.read_bytes()
                if _record_digest(data) != expected_hash:
                    raise ValueError("extracted RECORD hash mismatch")
        if extracted_files != set(expected) | {record_name}:
            raise ValueError("extracted wheel inventory mismatch")
        temp_path.rename(destination)
    except Exception:
        _cleanup_owned_temp(temp_path)
        raise

    return audit


def _read_bounded_json(path: Path) -> object:
    _require_regular_file(path, "model JSON")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("model JSON exceeds the size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_strings(value: object) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, list):
        for item in value:
            strings.extend(_collect_strings(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            strings.append(str(key))
            strings.extend(_collect_strings(item))
    return strings


def _require_no_symlink_components(path: Path, anchor: Path) -> None:
    if not path.is_absolute() or not anchor.is_absolute():
        raise ValueError("model path and anchor must be absolute")
    relative = path.relative_to(anchor)
    current = anchor
    _require_regular_directory(current, "model anchor")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("model path contains a symlink component")


def inspect_model_root(
    root: Path,
    *,
    min_total_bytes: int = MIN_MODEL_BYTES,
    max_total_bytes: int = MAX_MODEL_BYTES,
    anchor: Path = Path("/kaggle/input"),
) -> dict[str, object]:
    """Reject links/special files and verify the expected official preset shape."""

    _require_no_symlink_components(root, anchor)
    _require_regular_directory(root, "Gemma 4 model root")
    inventory: dict[str, int] = {}
    for base, directories, files in os.walk(root, topdown=True, followlinks=False):
        base_path = Path(base)
        for name in directories:
            child = base_path / name
            if child.is_symlink() or not child.is_dir():
                raise ValueError("model directory contains an unsafe directory")
        for name in files:
            child = base_path / name
            _require_regular_file(child, "model member")
            relative = child.relative_to(root).as_posix()
            if child.suffix.lower() not in ALLOWED_MODEL_SUFFIXES:
                raise ValueError("unexpected model file type")
            inventory[relative] = child.stat().st_size

    if not EXPECTED_MODEL_FILES.issubset(inventory):
        raise ValueError("official Keras Gemma 4 preset inventory is incomplete")
    if not 1 <= len(inventory) <= MAX_MODEL_FILES:
        raise ValueError("unexpected model file count")
    total_bytes = sum(inventory.values())
    if not min_total_bytes <= total_bytes <= max_total_bytes:
        raise ValueError("unexpected Gemma 4 preset size")

    config = _read_bounded_json(root / "config.json")
    task = _read_bounded_json(root / "task.json")
    weights = _read_bounded_json(root / "model.weights.json")
    config_strings = _collect_strings(config)
    task_strings = _collect_strings(task)
    if not any("Gemma4Backbone" in value for value in config_strings):
        raise ValueError("model config is not a Gemma4Backbone preset")
    if not any("Gemma4CausalLM" in value for value in task_strings):
        raise ValueError("task config is not a Gemma4CausalLM preset")
    referenced_shards = {
        PurePosixPath(value).name
        for value in _collect_strings(weights)
        if value.endswith(".weights.h5")
    }
    expected_shards = {"model_00000.weights.h5", "model_00001.weights.h5"}
    if referenced_shards != expected_shards:
        raise ValueError("model weight shard manifest mismatch")

    inventory_material = "".join(
        f"{name}\0{inventory[name]}\n" for name in sorted(inventory)
    ).encode("utf-8")
    return {
        "file_count": len(inventory),
        "inventory_sha256": hashlib.sha256(inventory_material).hexdigest(),
        "total_bytes": total_bytes,
    }


def configure_import_environment() -> None:
    if any(
        name == "keras_hub" or name.startswith("keras_hub.")
        for name in sys.modules
    ):
        raise RuntimeError("an older keras_hub was imported before vendoring")
    required = {
        "KERAS_BACKEND": "jax",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for key, value in required.items():
        existing = os.environ.get(key)
        if existing is not None and existing.lower() != value:
            raise RuntimeError(f"conflicting pre-import environment: {key}")
        os.environ[key] = value


def _normalize_generated_text(output: object) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)):
        if len(output) == 1 and isinstance(output[0], str):
            return output[0]
        raise ValueError("unexpected generated sequence shape")
    if isinstance(output, dict):
        if set(output) == {"prompts"}:
            prompts = output["prompts"]
            if isinstance(prompts, str):
                return prompts
            if (
                isinstance(prompts, (list, tuple))
                and len(prompts) == 1
                and isinstance(prompts[0], str)
            ):
                return prompts[0]
        raise ValueError("unexpected generated mapping shape")
    raise ValueError("unexpected generated output type")


def load_and_generate(model_root: Path, vendor_root: Path) -> dict[str, object]:
    """Import only the verified vendor, require P100, and run one short decode."""

    if sys.path[0] != str(vendor_root):
        sys.path.insert(0, str(vendor_root))

    import jax
    import keras
    import keras_hub

    module_file = Path(keras_hub.__file__).resolve(strict=True)
    vendor_resolved = vendor_root.resolve(strict=True)
    if not module_file.is_relative_to(vendor_resolved):
        raise RuntimeError("keras_hub was not imported from the verified vendor")
    version = importlib_metadata.version("keras-hub")
    if version != EXPECTED_KERAS_HUB_VERSION:
        raise RuntimeError("unexpected vendored keras_hub version")
    distribution_root = Path(
        importlib_metadata.distribution("keras-hub").locate_file("")
    ).resolve(strict=True)
    if not distribution_root.is_relative_to(vendor_resolved):
        raise RuntimeError("keras_hub distribution metadata is not vendored")
    if keras.backend.backend() != "jax":
        raise RuntimeError("Keras backend is not JAX")

    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError("smoke requires exactly one Kaggle GPU")
    device_kind = str(getattr(devices[0], "device_kind", ""))
    if "P100" not in device_kind.upper():
        raise RuntimeError("smoke requires a P100 GPU")

    model = keras_hub.models.Gemma4CausalLM.from_preset(
        str(model_root),
        dtype="float16",
    )
    model.compile(sampler="greedy")
    output = model.generate(
        {"prompts": [PROMPT]},
        max_length=MAX_LENGTH,
        strip_prompt=True,
    )
    generated = _normalize_generated_text(output).strip()
    generated_bytes = generated.encode("utf-8")
    if not generated_bytes:
        raise RuntimeError("Gemma 4 produced an empty completion")
    if len(generated_bytes) > MAX_GENERATED_UTF8_BYTES:
        raise RuntimeError("Gemma 4 completion exceeds the smoke bound")

    return {
        "backend": "jax",
        "generated_codepoints": len(generated),
        "generated_sha256": hashlib.sha256(generated_bytes).hexdigest(),
        "generated_utf8_bytes": len(generated_bytes),
        "gpu": "P100",
        "keras_hub_version": version,
        "max_length": MAX_LENGTH,
        "sampler": "greedy",
        "tensorflow_gpu_growth": True,
        "weight_dtype": "float16",
    }


def _write_report(report: dict[str, object]) -> None:
    if REPORT_PATH.exists() or REPORT_PATH.is_symlink():
        raise ValueError("smoke report destination already exists")
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("smoke report temporary path already exists")
    temporary.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REPORT_PATH)


def _write_journal(stage: str, details: dict[str, object]) -> None:
    """Atomically preserve the last safe stage without generated text."""

    if not stage or any(key not in {"error_type", "generated_sha256", "model_source", "status", "wheel_sha256"} for key in details):
        raise ValueError("unsafe smoke journal payload")
    payload = {
        "details": details,
        "schema_version": "1.0",
        "stage": stage,
    }
    temporary = JOURNAL_PATH.with_suffix(".json.tmp")
    if JOURNAL_PATH.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise ValueError("unsafe smoke journal destination")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(JOURNAL_PATH)


def main() -> None:
    model_source = "keras/gemma4/keras/gemma4_instruct_2b/2"
    _write_journal("preflight_started", {"model_source": model_source})
    try:
        _require_regular_directory(WHEEL_DATASET_ROOT, "verified wheel dataset")
        wheel_report = audit_wheel(WHEEL_PATH)
        model_report = inspect_model_root(MODEL_ROOT)
        extract_verified_wheel(WHEEL_PATH, VENDOR_ROOT)
        configure_import_environment()
        _write_journal(
            "model_load_started",
            {
                "model_source": model_source,
                "wheel_sha256": EXPECTED_WHEEL_SHA256,
            },
        )
        runtime_report = load_and_generate(MODEL_ROOT, VENDOR_ROOT)
        report = {
            "internet_enabled": False,
            "model": model_report,
            "model_source": model_source,
            "schema_version": "1.0",
            "status": "PASS",
            "runtime": runtime_report,
            "wheel": wheel_report,
            "wheel_dataset_source": (
                "YOUR_KAGGLE_USERNAME/verified-keras-hub-028-gemma4"
            ),
        }
        _write_report(report)
        _write_journal(
            "complete",
            {
                "generated_sha256": runtime_report["generated_sha256"],
                "status": "PASS",
            },
        )
    except Exception as exc:
        _write_journal("failed", {"error_type": type(exc).__name__})
        raise
    print("PASS: verified Gemma 4 smoke report written")


if __name__ == "__main__":
    main()
