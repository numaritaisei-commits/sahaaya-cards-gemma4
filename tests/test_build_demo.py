import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import build_demo
import validate_project as validator


def _test_bundle(fixture, headline="Validated test headline"):
    notice_id = fixture["notice_id"]
    ledger = [
        {
            "fact_id": "F1",
            "kind": "warning",
            "value": "Test-only normalized fact",
            "source_quote": fixture["title"],
        }
    ]
    cards = []
    for language in ("en", "hi", "ta"):
        cards.append(
            {
                "language": language,
                "headline": headline,
                "source_fact_ids": ["F1"],
                "actions": [{"text": "Test-only action", "fact_ids": ["F1"]}],
                "do_not_infer": ["Test-only limitation"],
            }
        )
    checks = [
        {
            "claim_path": "fact_ledger[0].value",
            "supported": True,
            "fact_ids": ["F1"],
            "explanation": "Test-only check",
        }
    ]
    for card_index in range(3):
        checks.extend(
            [
                {
                    "claim_path": f"cards[{card_index}].headline",
                    "supported": True,
                    "fact_ids": ["F1"],
                    "explanation": "Test-only check",
                },
                {
                    "claim_path": f"cards[{card_index}].actions[0].text",
                    "supported": True,
                    "fact_ids": ["F1"],
                    "explanation": "Test-only check",
                },
                {
                    "claim_path": f"cards[{card_index}].do_not_infer[0]",
                    "supported": True,
                    "fact_ids": [],
                    "explanation": "Test-only boundary check",
                },
            ]
        )
    return {
        "generator": {
            "notice_id": notice_id,
            "fact_ledger": ledger,
            "cards": cards,
            "uncertainties": [],
        },
        "verifier": {
            "notice_id": notice_id,
            "verdict": "PASS",
            "checks": checks,
            "unsupported_claims": [],
            "language_warnings": [],
            "safety_note": "Test-only safety note",
        },
    }


def _valid_test_artifact(headline="Validated test headline"):
    notices = []
    for fixture_path in validator.FIXTURE_PATHS:
        fixture = validator.load_json(fixture_path)
        parsed = _test_bundle(fixture, headline=headline)
        notices.append(
            {
                "notice_id": fixture["notice_id"],
                "source_sha256": hashlib.sha256(
                    json.dumps(fixture, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "prompts": {"generator": "test prompt", "verifier": "test prompt"},
                "raw_final_answers": {
                    "generator": "TEST-ONLY RAW VALUE THAT MUST NOT BE RENDERED",
                    "verifier": "TEST-ONLY RAW VALUE THAT MUST NOT BE RENDERED",
                },
                "parsed": parsed,
                "timing_seconds": {"generator": 1.0, "verifier": 2.0, "total": 3.0},
                "validation": {"passed": True, "errors": []},
            }
        )
    return {
        "schema_version": "1.0",
        "project": "Sahaaya Cards",
        "generated_at_utc": "2030-01-01T00:00:00+00:00",
        "model_ref": validator.MODEL_REF,
        "model_path": validator.MODEL_PATH,
        "run_configuration": {
            "framework": "keras_hub",
            "backend": "jax",
            "weight_dtype": "float16",
            "keras_hub_version": "0.28.0",
            "wheel_dataset_ref": validator.WHEEL_DATASET_REF,
            "wheel_sha256": validator.WHEEL_SHA256,
            "max_length": 2048,
            "sampler": "greedy",
            "strip_prompt": True,
            "tensorflow_gpu_growth": True,
            "internet_enabled": False,
            "external_apis": False,
        },
        "safety_limitations": ["one", "two", "three", "four"],
        "notices": notices,
    }


class OfflineDemoTests(unittest.TestCase):
    def test_missing_runtime_artifact_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "demo" / "index.html"
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(root / "missing.json", output)
            self.assertFalse(output.exists())

    def test_invalid_runtime_artifact_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(source, output)
            self.assertFalse(output.exists())

    def test_duplicate_json_key_refuses_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(
                '{"schema_version":"1.0","schema_version":"1.0"}',
                encoding="utf-8",
            )
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(source, output)
            self.assertFalse(output.exists())

    def test_more_than_six_facts_are_rejected(self):
        artifact = _valid_test_artifact()
        ledger = artifact["notices"][0]["parsed"]["generator"]["fact_ledger"]
        for index in range(2, 8):
            ledger.append(
                {
                    "fact_id": f"F{index}",
                    "kind": "warning",
                    "value": "Test-only extra fact",
                    "source_quote": ledger[0]["source_quote"],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(source, output)
            self.assertFalse(output.exists())

    def test_multiple_actions_are_rejected(self):
        artifact = _valid_test_artifact()
        actions = artifact["notices"][0]["parsed"]["generator"]["cards"][0]["actions"]
        actions.append(dict(actions[0]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(source, output)
            self.assertFalse(output.exists())

    def test_action_over_35_words_is_rejected(self):
        artifact = _valid_test_artifact()
        artifact["notices"][0]["parsed"]["generator"]["cards"][0]["actions"][0][
            "text"
        ] = "word " * 36
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(source, output)
            self.assertFalse(output.exists())

    def test_valid_test_artifact_renders_offline_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(
                json.dumps(_valid_test_artifact(), ensure_ascii=False), encoding="utf-8"
            )
            rendered = build_demo.render_demo(source, output)
            page = rendered.read_text(encoding="utf-8")
            self.assertIn("SYN-FLOOD-001", page)
            self.assertIn("SYN-WATER-002", page)
            self.assertIn("English", page)
            self.assertIn("हिन्दी", page)
            self.assertIn("தமிழ்", page)
            self.assertIn("Verifier PASS", page)
            self.assertNotIn("TEST-ONLY RAW VALUE THAT MUST NOT BE RENDERED", page)
            self.assertNotIn("<script", page.lower())
            self.assertNotIn("http://", page.lower())
            self.assertNotIn("https://", page.lower())

    def test_project_validator_accepts_external_deep_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "demo_results.json"
            source.write_text(
                json.dumps(_valid_test_artifact(), ensure_ascii=False), encoding="utf-8"
            )
            self.assertEqual(validator.validate_runtime_semantics_path(source), [])

    def test_project_validator_rejects_false_pass_with_tampered_quote(self):
        artifact = _valid_test_artifact()
        artifact["notices"][0]["parsed"]["generator"]["fact_ledger"][0][
            "source_quote"
        ] = "THIS QUOTE IS NOT IN THE TRUSTED FIXTURE"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "demo_results.json"
            source.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
            errors = validator.validate_runtime_semantics_path(source)
            self.assertEqual(errors, ["runtime artifact failed deep evidence validation"])

    def test_model_derived_html_is_escaped(self):
        hostile = "<img src=x onerror=alert(1)>"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(
                json.dumps(_valid_test_artifact(headline=hostile), ensure_ascii=False),
                encoding="utf-8",
            )
            build_demo.render_demo(source, output)
            page = output.read_text(encoding="utf-8")
            self.assertNotIn(hostile, page)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", page)

    def test_incomplete_verifier_coverage_is_rejected(self):
        artifact = _valid_test_artifact()
        artifact["notices"][0]["parsed"]["verifier"]["checks"].pop()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(source, output)
            self.assertFalse(output.exists())

    def test_duplicate_notice_is_rejected(self):
        artifact = _valid_test_artifact()
        artifact["notices"].append(artifact["notices"][0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "demo_results.json"
            output = root / "demo" / "index.html"
            source.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(build_demo.DemoRefusal):
                build_demo.render_demo(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
