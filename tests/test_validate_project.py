import hashlib
import unittest

import validate_project as validator


def valid_runtime_artifact():
    notices = []
    for notice_id in sorted(validator.EXPECTED_NOTICE_IDS):
        notices.append(
            {
                "notice_id": notice_id,
                "source_sha256": hashlib.sha256(notice_id.encode("utf-8")).hexdigest(),
                "prompts": {"generator": "generator prompt", "verifier": "verifier prompt"},
                "raw_final_answers": {"generator": "{}", "verifier": "{}"},
                "parsed": {"generator": {}, "verifier": {}},
                "timing_seconds": {"generator": 1.0, "verifier": 1.0, "total": 2.0},
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


class ProjectValidatorTests(unittest.TestCase):
    def test_project_static_validation_passes(self):
        report = validator.run_static_checks()
        self.assertEqual(report["static_status"], "GO", report["errors"])

    def test_valid_runtime_schema_passes(self):
        self.assertEqual(validator.validate_runtime_artifact(valid_runtime_artifact()), [])

    def test_wrong_model_ref_fails(self):
        artifact = valid_runtime_artifact()
        artifact["model_ref"] = "untrusted/model"
        errors = validator.validate_runtime_artifact(artifact)
        self.assertIn("runtime artifact model_ref mismatch", errors)

    def test_only_verified_gemma4_route_is_present(self):
        notebook = validator.load_json(validator.ROOT / validator.NOTEBOOK_NAME)
        notebook_errors, notebook_source = validator.validate_notebook(notebook)
        self.assertEqual(notebook_errors, [])
        self.assertIn(validator.MODEL_REF, notebook_source)
        self.assertIn(validator.MODEL_PATH, notebook_source)
        self.assertIn("keras_hub.models.Gemma4CausalLM.from_preset", notebook_source)
        self.assertIn(validator.WHEEL_DATASET_REF, notebook_source)
        self.assertIn(validator.WHEEL_SHA256, notebook_source)
        self.assertIn("record_entries_verified", notebook_source)
        self.assertIn("record_bytes = archive.read(record_name)", notebook_source)
        self.assertIn("if relative == record_name:", notebook_source)
        self.assertIn("set(expected) | {record_name}", notebook_source)
        self.assertIn("_read_bounded_json", notebook_source)
        self.assertIn("TF_FORCE_GPU_ALLOW_GROWTH", notebook_source)
        self.assertIn("MAX_LENGTH = 2048", notebook_source)
        self.assertIn("_reject_duplicate_object_pairs", notebook_source)
        self.assertIn("write_runtime_journal", notebook_source)
        self.assertIn("runtime_journal.json", notebook_source)
        self.assertIn("fact_ledger must contain one to six facts", notebook_source)
        self.assertIn("actions must contain exactly one item", notebook_source)
        self.assertIn("generated text logging disabled", notebook_source)
        self.assertIn("strip_prompt=True", notebook_source)
        self.assertIn(
            'prompt = "<|turn>user\\n" + user_text + "<turn|>\\n<|turn>model\\n"',
            notebook_source,
        )
        self.assertIn('{"prompts": [prompt]}', notebook_source)
        self.assertIn("len(text.split()) > 35", notebook_source)
        self.assertIn("fact_ledger[{fact_index}].value", notebook_source)
        self.assertIn("do_not_infer[{limit_index}]", notebook_source)
        for token in validator.FORBIDDEN_NOTEBOOK_TOKENS:
            self.assertNotIn(token, notebook_source)
        invalid_values = ("untrusted/model", "keras/gemma3/keras/gemma3_instruct_1b/3")
        for retired in invalid_values:
            artifact = valid_runtime_artifact()
            artifact["model_ref"] = retired
            artifact["model_path"] = retired
            errors = validator.validate_runtime_artifact(artifact)
            self.assertIn("runtime artifact model_ref mismatch", errors)
            self.assertIn("runtime artifact model_path mismatch", errors)

    def test_public_cover_is_bounded_canonical_png(self):
        self.assertEqual(validator.validate_binary_assets(), [])

    def test_publication_scan_rejects_pii_secret_and_local_path(self):
        sample = (
            "/" + "Users" + "/example/private.txt\n"
            "contact" + "@" + "example.test\n"
            "+81" + " 80" + " 1234" + " 5678\n"
            "ghp_" + "abcdefghijklmnopqrstuvwxyz" + "12345" + "67890\n"
        )
        errors = validator.scan_publication_text("sample.txt", sample)
        self.assertTrue(any("host-specific user path" in error for error in errors))
        self.assertTrue(any("email address" in error for error in errors))
        self.assertTrue(any("phone-like value" in error for error in errors))
        self.assertTrue(any("credential-like value" in error for error in errors))

    def test_failed_notice_blocks_publication(self):
        artifact = valid_runtime_artifact()
        artifact["notices"][0]["validation"] = {
            "passed": False,
            "errors": ["unsupported claim"],
        }
        errors = validator.validate_runtime_artifact(artifact, require_pass=True)
        self.assertTrue(any("did not pass" in error for error in errors))
        self.assertTrue(any("contains deterministic runtime errors" in error for error in errors))

    def test_network_import_is_rejected(self):
        errors = validator.scan_code_security("import requests\nrequests.get('example')\n")
        self.assertTrue(any("banned import" in error for error in errors))
        self.assertTrue(any("banned attribute call" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
