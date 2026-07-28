"""Fail-closed static audit for the pinned pure-Python KerasHub wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_FILENAME = "keras_hub-0.28.0-py3-none-any.whl"
EXPECTED_SHA256 = "a28cb601f7fffb7f28add1bae8110459fc3ac7d9e2159453dfbda9e97271fc87"
EXPECTED_TOP_LEVEL = {"keras_hub", "keras_hub-0.28.0.dist-info"}
FORBIDDEN_SUFFIXES = {
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
ALLOWED_FILE_SUFFIXES = {".py", ".txt"}
ALLOWED_EXTENSIONLESS_NAMES = {"METADATA", "RECORD", "WHEEL"}
MAX_ARCHIVE_BYTES = 2_000_000
MAX_UNCOMPRESSED_BYTES = 12_000_000
MAX_MEMBER_BYTES = 2_000_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return encoded.rstrip(b"=").decode("ascii")


def inspect_archive_members(archive: zipfile.ZipFile) -> dict[str, Any]:
    """Validate every ZIP member before metadata reads or extraction."""
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if not names:
        raise ValueError("empty wheel")
    if len(names) != len(set(names)):
        raise ValueError("duplicate ZIP member")

    canonical_names: set[str] = set()
    file_names: set[str] = set()
    top_levels: set[str] = set()
    total_size = 0
    native_payloads = 0
    symlinks = 0
    for info in infos:
        name = info.filename
        member = PurePosixPath(name)
        if not name or name.startswith("/") or member.is_absolute() or not member.parts:
            raise ValueError("unsafe absolute or empty ZIP member path")
        if ".." in member.parts or "\\" in name or "\x00" in name:
            raise ValueError("malformed ZIP member path")
        is_directory = info.is_dir()
        canonical_name = member.as_posix() + ("/" if is_directory else "")
        if name != canonical_name or canonical_name in canonical_names:
            raise ValueError("non-canonical or colliding ZIP member path")
        canonical_names.add(canonical_name)

        mode = (info.external_attr >> 16) & 0o177777
        file_type = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode):
            symlinks += 1
            raise ValueError("symlink member is forbidden")
        if info.flag_bits & 0x1:
            raise ValueError("encrypted ZIP member is forbidden")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError("unsupported ZIP compression method")
        if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise ValueError("ZIP member exceeds size limit")

        if is_directory:
            if (
                file_type not in {0, stat.S_IFDIR}
                or info.file_size != 0
                or info.compress_size != 0
            ):
                raise ValueError("malformed ZIP directory entry")
        else:
            if file_type not in {0, stat.S_IFREG}:
                raise ValueError("ZIP special-file member is forbidden")
            if mode & 0o111:
                raise ValueError("executable ZIP member is forbidden")
            suffix = member.suffix.lower()
            if suffix in FORBIDDEN_SUFFIXES:
                native_payloads += 1
            elif (
                suffix not in ALLOWED_FILE_SUFFIXES
                and member.name not in ALLOWED_EXTENSIONLESS_NAMES
            ):
                raise ValueError("ZIP member suffix is not allowlisted")
            file_names.add(name)

        total_size += info.file_size
        top_levels.add(member.parts[0])

    if native_payloads:
        raise ValueError("native or executable payload is forbidden")
    if total_size > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("wheel expands beyond size limit")
    if top_levels != EXPECTED_TOP_LEVEL:
        raise ValueError("unexpected top-level wheel inventory")
    return {
        "file_names": file_names,
        "member_count": len(infos),
        "native_payloads": native_payloads,
        "symlinks": symlinks,
        "top_levels": top_levels,
        "uncompressed_bytes": total_size,
    }


def audit(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("wheel must be a regular non-symlink file")
    if path.name != EXPECTED_FILENAME:
        raise ValueError("unexpected wheel filename")
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("wheel archive exceeds size limit")
    digest = file_sha256(path)
    if digest != EXPECTED_SHA256:
        raise ValueError("wheel SHA-256 mismatch")

    with zipfile.ZipFile(path) as archive:
        inventory = inspect_archive_members(archive)
        file_names = inventory["file_names"]

        dist_info = "keras_hub-0.28.0.dist-info"
        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        wheel_meta = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
        record_name = f"{dist_info}/RECORD"
        record_text = archive.read(record_name).decode("utf-8")

        if "Name: keras-hub\n" not in metadata or "Version: 0.28.0\n" not in metadata:
            raise ValueError("wheel package identity mismatch")
        if "Root-Is-Purelib: true\n" not in wheel_meta:
            raise ValueError("wheel is not declared pure Python")
        if "Tag: py3-none-any\n" not in wheel_meta:
            raise ValueError("unexpected wheel compatibility tag")

        record_rows = list(csv.reader(io.StringIO(record_text)))
        if any(len(row) != 3 for row in record_rows):
            raise ValueError("malformed RECORD row")
        record_names = [row[0] for row in record_rows]
        if len(record_names) != len(set(record_names)):
            raise ValueError("duplicate RECORD row")
        record_paths = {row[0] for row in record_rows}
        if record_paths != file_names:
            raise ValueError("RECORD inventory mismatch")
        for row in record_rows:
            name, hash_field, size_field = row
            if name == record_name:
                if hash_field or size_field:
                    raise ValueError("RECORD self-entry must be unhashed")
                continue
            if not hash_field.startswith("sha256="):
                raise ValueError("non-SHA256 or missing RECORD hash")
            data = archive.read(name)
            if hash_field.removeprefix("sha256=") != record_digest(data):
                raise ValueError("RECORD member hash mismatch")
            if size_field != str(len(data)):
                raise ValueError("RECORD member size mismatch")

    return {
        "archive_bytes": path.stat().st_size,
        "member_count": inventory["member_count"],
        "native_payloads": inventory["native_payloads"],
        "record_entries_verified": len(record_rows) - 1,
        "sha256": digest,
        "status": "PASS",
        "symlinks": inventory["symlinks"],
        "uncompressed_bytes": inventory["uncompressed_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.wheel), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
