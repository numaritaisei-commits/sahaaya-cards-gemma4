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


EXPECTED_FILENAME = "keras_hub-0.28.0-py3-none-any.whl"
EXPECTED_SHA256 = "a28cb601f7fffb7f28add1bae8110459fc3ac7d9e2159453dfbda9e97271fc87"
EXPECTED_TOP_LEVEL = {"keras_hub", "keras_hub-0.28.0.dist-info"}
FORBIDDEN_SUFFIXES = {
    ".a",
    ".dylib",
    ".dll",
    ".exe",
    ".node",
    ".o",
    ".pyd",
    ".so",
}
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
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        if not names:
            raise ValueError("empty wheel")

        total_size = 0
        top_levels: set[str] = set()
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or not member.parts:
                raise ValueError("unsafe ZIP member path")
            if "\\" in info.filename or "\x00" in info.filename:
                raise ValueError("malformed ZIP member path")
            mode = (info.external_attr >> 16) & 0o177777
            if stat.S_ISLNK(mode):
                raise ValueError("symlink member is forbidden")
            if member.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise ValueError("native or executable payload is forbidden")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError("ZIP member exceeds size limit")
            total_size += info.file_size
            top_levels.add(member.parts[0])

        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("wheel expands beyond size limit")
        if top_levels != EXPECTED_TOP_LEVEL:
            raise ValueError("unexpected top-level wheel inventory")

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
        record_paths = {row[0] for row in record_rows}
        if record_paths != set(names):
            raise ValueError("RECORD inventory mismatch")
        for row in record_rows:
            if len(row) != 3:
                raise ValueError("malformed RECORD row")
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
        "member_count": len(names),
        "native_payloads": 0,
        "record_entries_verified": len(record_rows) - 1,
        "sha256": digest,
        "status": "PASS",
        "symlinks": 0,
        "uncompressed_bytes": total_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.wheel), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
