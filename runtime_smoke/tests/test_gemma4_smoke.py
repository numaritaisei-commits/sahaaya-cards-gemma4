from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gemma4_smoke as smoke
import validate_smoke as validator


def digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def make_wheel(path: Path, extra=None, bad_record=False) -> str:
    files = {
        "keras_hub/__init__.py": b"__version__ = '0.28.0'\n",
        "keras_hub-0.28.0.dist-info/METADATA": (
            b"Name: keras-hub\nVersion: 0.28.0\nRequires-Python: >=3.11\n"
            b"Requires-Dist: keras>=3.13\n"
        ),
        "keras_hub-0.28.0.dist-info/WHEEL": b"Root-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files.update(extra or {})
    record_name = "keras_hub-0.28.0.dist-info/RECORD"
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for name, data in files.items():
        writer.writerow([name, "sha256=" + digest(data), len(data)])
    writer.writerow([record_name, "", ""])
    files[record_name] = out.getvalue().encode()
    if bad_record:
        files[record_name] = files[record_name].replace(b"sha256=", b"sha256=bad", 1)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)
    return smoke.file_sha256(path)


class SmokeTests(unittest.TestCase):
    def valid_output_records(self):
        generated_sha256 = "a" * 64
        report = {
            "internet_enabled": False,
            "model": {
                "file_count": 12,
                "inventory_sha256": "b" * 64,
                "total_bytes": 11_000_000_000,
            },
            "model_source": validator.EXPECTED_MODEL_SOURCE,
            "runtime": {
                "backend": "jax",
                "generated_codepoints": 5,
                "generated_sha256": generated_sha256,
                "generated_utf8_bytes": 5,
                "gpu": "P100",
                "keras_hub_version": "0.28.0",
                "max_length": 96,
                "sampler": "greedy",
                "tensorflow_gpu_growth": True,
                "weight_dtype": "float16",
            },
            "schema_version": "1.0",
            "status": "PASS",
            "wheel": dict(validator.EXPECTED_WHEEL_REPORT),
            "wheel_dataset_source": validator.EXPECTED_DATASET_SOURCE,
        }
        journal = {
            "details": {"generated_sha256": generated_sha256, "status": "PASS"},
            "schema_version": "1.0",
            "stage": "complete",
        }
        return report, journal

    def wheel_case(self, extra=None, bad_record=False):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / smoke.EXPECTED_WHEEL_FILENAME
        sha = make_wheel(path, extra, bad_record)
        return temp, path, sha

    def test_valid_wheel(self):
        temp, path, sha = self.wheel_case()
        self.addCleanup(temp.cleanup)
        self.assertEqual(smoke.audit_wheel(path, expected_sha256=sha)["sha256"], sha)

    def test_verified_extraction_preserves_record_self_entry(self):
        temp, path, sha = self.wheel_case()
        self.addCleanup(temp.cleanup)
        working_root = Path(temp.name) / "working"
        working_root.mkdir()
        destination = working_root / "vendor"
        with mock.patch.object(smoke, "WORKING_ROOT", working_root):
            report = smoke.extract_verified_wheel(
                path,
                destination,
                expected_sha256=sha,
            )
        record = destination / "keras_hub-0.28.0.dist-info" / "RECORD"
        self.assertTrue(record.is_file())
        self.assertEqual(report["record_entries_verified"], 3)

    def test_sha_mismatch_rejected(self):
        temp, path, _ = self.wheel_case()
        self.addCleanup(temp.cleanup)
        with self.assertRaises(ValueError):
            smoke.audit_wheel(path, expected_sha256="0" * 64)

    def test_native_payload_rejected(self):
        temp, path, sha = self.wheel_case({"keras_hub/bad.so": b"x"})
        self.addCleanup(temp.cleanup)
        with self.assertRaises(ValueError):
            smoke.audit_wheel(path, expected_sha256=sha)

    def test_traversal_rejected(self):
        temp, path, sha = self.wheel_case({"../escape.py": b"x"})
        self.addCleanup(temp.cleanup)
        with self.assertRaises(ValueError):
            smoke.audit_wheel(path, expected_sha256=sha)

    def test_bad_record_rejected(self):
        temp, path, sha = self.wheel_case(bad_record=True)
        self.addCleanup(temp.cleanup)
        with self.assertRaises(ValueError):
            smoke.audit_wheel(path, expected_sha256=sha)

    def test_model_inventory_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            anchor = Path(temp)
            root = anchor / "model"
            (root / "assets/tokenizer").mkdir(parents=True)
            (root / "config.json").write_text(json.dumps({"class_name": "Gemma4Backbone"}))
            (root / "task.json").write_text(json.dumps({"class_name": "Gemma4CausalLM"}))
            shards = ["model_00000.weights.h5", "model_00001.weights.h5"]
            (root / "model.weights.json").write_text(json.dumps(shards))
            (root / "assets/tokenizer/vocabulary.spm").write_bytes(b"spm")
            for name in shards:
                (root / name).write_bytes(b"weight")
            result = smoke.inspect_model_root(root, min_total_bytes=0, max_total_bytes=10_000, anchor=anchor)
            self.assertEqual(result["file_count"], 6)
            (root / "bad.json").symlink_to(root / "config.json")
            with self.assertRaises(ValueError):
                smoke.inspect_model_root(root, min_total_bytes=0, max_total_bytes=10_000, anchor=anchor)

    def test_generated_output_shapes(self):
        self.assertEqual(smoke._normalize_generated_text(["ok"]), "ok")
        self.assertEqual(smoke._normalize_generated_text({"prompts": ["ok"]}), "ok")
        with self.assertRaises(ValueError):
            smoke._normalize_generated_text(["a", "b"])

    def test_journal_is_atomic_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Path(temp) / "journal.json"
            with mock.patch.object(smoke, "JOURNAL_PATH", journal):
                smoke._write_journal("preflight_started", {"status": "READY"})
                smoke._write_journal("failed", {"error_type": "RuntimeError"})
            payload = json.loads(journal.read_text(encoding="utf-8"))
            self.assertEqual(payload["stage"], "failed")
            self.assertEqual(payload["details"], {"error_type": "RuntimeError"})
            self.assertFalse(journal.with_suffix(".json.tmp").exists())

    def test_static_source_and_metadata(self):
        validator.validate_source(ROOT / "gemma4_smoke.py")
        validator.validate_metadata(ROOT / "kernel-metadata.example.json")

    def test_report_and_journal_are_bound_and_complete(self):
        report, journal = self.valid_output_records()
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            journal_path = Path(temp) / "journal.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            journal_path.write_text(json.dumps(journal), encoding="utf-8")
            digest = validator.validate_report(report_path)
            validator.validate_journal(journal_path, digest)

    def test_false_pass_with_wrong_sources_is_rejected(self):
        report, _ = self.valid_output_records()
        report["wheel_dataset_source"] = "untrusted/source"
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                validator.validate_report(report_path)

    def test_mismatched_journal_digest_is_rejected(self):
        report, journal = self.valid_output_records()
        journal["details"]["generated_sha256"] = "c" * 64
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            journal_path = Path(temp) / "journal.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            journal_path.write_text(json.dumps(journal), encoding="utf-8")
            digest = validator.validate_report(report_path)
            with self.assertRaises(ValueError):
                validator.validate_journal(journal_path, digest)

    def test_duplicate_report_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            report_path.write_text('{"status":"PASS","status":"PASS"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                validator.validate_report(report_path)


if __name__ == "__main__":
    unittest.main()
