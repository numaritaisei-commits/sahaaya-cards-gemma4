import hashlib
import json
import re
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
                "token_budgets": {
                    "generator": {
                        "completion_budget_tokens": 1948,
                        "max_length": 2048,
                        "minimum_completion_tokens": 512,
                        "prompt_tokens": 100,
                    },
                    "verifier": {
                        "completion_budget_tokens": 2872,
                        "max_length": 3072,
                        "minimum_completion_tokens": 512,
                        "prompt_tokens": 200,
                    },
                },
                "timing_seconds": {"generator": 1.0, "verifier": 1.0, "total": 2.0},
                "validation": {"passed": True, "errors": []},
            }
        )
    return {
        "schema_version": "1.0",
        "project": "Sahaaya Cards",
        "generated_at_utc": "2030-01-01T00:00:00+00:00",
        "status": "PASS",
        "model_ref": validator.MODEL_REF,
        "model_path": validator.MODEL_PATH,
        "run_configuration": {
            "framework": "keras_hub",
            "backend": "torch",
            "weight_dtype": "float16",
            "keras_hub_version": "0.28.0",
            "wheel_dataset_ref": f"testowner/{validator.WHEEL_DATASET_SLUG}",
            "wheel_sha256": validator.WHEEL_SHA256,
            "generator_max_length": 2048,
            "verifier_max_length": 3072,
            "minimum_completion_tokens": 512,
            "sampler": "greedy",
            "strip_prompt": True,
            "gpu": "T4x2",
            "tensorflow_gpu_visible": False,
            "internet_enabled": False,
            "external_apis": False,
        },
        "runtime_provenance": {
            "backend": "torch",
            "explicit_model_dtype": "float16",
            "global_policy": "float16",
            "gpu": "T4x2",
            "gpu_count": 2,
            "keras_hub_version": "0.28.0",
            "load_seconds": 120.0,
            "memory_after_load": {
                "allocated_bytes": 9_000_000_000,
                "free_bytes": 5_000_000_000,
                "peak_allocated_bytes": 9_000_000_000,
                "reserved_bytes": 9_500_000_000,
                "total_bytes": 15_000_000_000,
            },
            "memory_before_load": {
                "allocated_bytes": 0,
                "free_bytes": 14_000_000_000,
                "peak_allocated_bytes": 0,
                "reserved_bytes": 0,
                "total_bytes": 15_000_000_000,
            },
            "model": {
                "file_count": 6,
                "inventory_sha256": "1" * 64,
                "total_bytes": 11_000_000_000,
            },
            "model_variable_bytes": 9_000_000_000,
            "official_weights_loaded": True,
            "tensorflow_gpu_visible": False,
            "torch_version": "2.8.0",
            "variable_bytes_by_device": {"cuda:0": 9_000_000_000},
            "variable_bytes_by_dtype": {"float16": 9_000_000_000},
            "weight_dtype": "float16",
            "wheel": {
                "archive_bytes": 1_525_623,
                "member_count": 766,
                "record_entries_verified": 765,
                "sha256": validator.WHEEL_SHA256,
            },
        },
        "safety_limitations": ["one", "two", "three", "four"],
        "failures": [],
        "notices": notices,
    }


class ProjectValidatorTests(unittest.TestCase):
    def test_project_static_validation_passes(self):
        report = validator.run_static_checks()
        self.assertEqual(report["static_status"], "GO", report["errors"])

    def test_valid_runtime_schema_passes(self):
        self.assertEqual(validator.validate_runtime_artifact(valid_runtime_artifact()), [])

    def test_clone_owner_replacement_keeps_notebook_metadata_and_runtime_compatible(self):
        owner = "testowner"
        notebook_text = (
            validator.ROOT / validator.NOTEBOOK_NAME
        ).read_text(encoding="utf-8").replace(validator.OWNER_PLACEHOLDER, owner)
        notebook = json.loads(notebook_text)
        notebook_errors, _ = validator.validate_notebook(notebook)
        self.assertEqual(notebook_errors, [])

        metadata_text = (
            validator.ROOT / validator.KERNEL_METADATA_NAME
        ).read_text(encoding="utf-8").replace(validator.OWNER_PLACEHOLDER, owner)
        self.assertEqual(validator.validate_metadata(json.loads(metadata_text)), [])

        artifact = valid_runtime_artifact()
        artifact["run_configuration"]["wheel_dataset_ref"] = (
            f"{owner}/{validator.WHEEL_DATASET_SLUG}"
        )
        self.assertEqual(validator.validate_runtime_artifact(artifact), [])

    def test_public_placeholder_cannot_claim_runtime_pass(self):
        artifact = valid_runtime_artifact()
        artifact["run_configuration"]["wheel_dataset_ref"] = validator.WHEEL_DATASET_REF
        errors = validator.validate_runtime_artifact(artifact, require_pass=True)
        self.assertIn("runtime wheel Dataset reference is invalid", errors)

    def test_schema_and_validator_share_safe_runtime_owner_contract(self):
        schema = json.loads(
            (validator.ROOT / "schemas" / "demo_results.schema.json").read_text(
                encoding="utf-8"
            )
        )
        ref_contract = schema["properties"]["run_configuration"]["properties"][
            "wheel_dataset_ref"
        ]
        self.assertEqual(set(ref_contract), {"type", "pattern"})
        self.assertEqual(ref_contract["type"], "string")
        pattern = ref_contract["pattern"]

        safe_ref = f"safe-owner_2/{validator.WHEEL_DATASET_SLUG}"
        self.assertIsNotNone(re.fullmatch(pattern, safe_ref))
        self.assertEqual(
            validator._wheel_dataset_owner(safe_ref, allow_placeholder=False),
            "safe-owner_2",
        )
        artifact = valid_runtime_artifact()
        artifact["run_configuration"]["wheel_dataset_ref"] = safe_ref
        self.assertEqual(validator.validate_runtime_artifact(artifact), [])

        unsafe_refs = (
            validator.WHEEL_DATASET_REF,
            f"Uppercase/{validator.WHEEL_DATASET_SLUG}",
            f"-leading/{validator.WHEEL_DATASET_SLUG}",
            f"owner/../{validator.WHEEL_DATASET_SLUG}",
            "owner/wrong-wheel-dataset",
        )
        for unsafe_ref in unsafe_refs:
            with self.subTest(unsafe_ref=unsafe_ref):
                self.assertIsNone(re.fullmatch(pattern, unsafe_ref))
                self.assertIsNone(
                    validator._wheel_dataset_owner(
                        unsafe_ref,
                        allow_placeholder=False,
                    )
                )

        readme = (validator.ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("replace the owner placeholder consistently", readme)
        self.assertIn("safe clone owner bound to the pinned Dataset slug", readme)

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
        self.assertIn("GENERATOR_MAX_LENGTH = 2048", notebook_source)
        self.assertIn("VERIFIER_MAX_LENGTH = 3072", notebook_source)
        self.assertIn("MIN_COMPLETION_TOKENS = 512", notebook_source)
        self.assertIn('"KERAS_BACKEND": "torch"', notebook_source)
        self.assertIn('tf.config.set_visible_devices([], "GPU")', notebook_source)
        self.assertIn('with keras.device("gpu:0")', notebook_source)
        self.assertIn("tokenizer prompt budget leaves fewer than 512 completion tokens", notebook_source)
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

    def test_public_notebook_discloses_failed_runtime_evidence(self):
        notebook = validator.load_json(validator.ROOT / validator.NOTEBOOK_NAME)
        status_note = "".join(notebook["cells"][0]["source"])
        for phrase in (
            "NOT RUNTIME-PROVEN",
            "V4",
            "V5",
            "V6",
            "not model-generated",
        ):
            self.assertIn(phrase, status_note)

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

    def test_partial_artifact_cannot_be_a_publication_pass(self):
        artifact = valid_runtime_artifact()
        artifact["status"] = "PARTIAL"
        artifact["failures"] = [{
            "error_type": "MemoryError",
            "failure_classification": "oom",
            "stage": "model_load",
        }]
        errors = validator.validate_runtime_artifact(artifact, require_pass=True)
        self.assertIn("runtime artifact status is not PASS", errors)
        self.assertIn("PASS runtime artifact must contain no failures", errors)

    def test_preflight_failure_artifact_remains_valid_diagnostic_evidence(self):
        artifact = valid_runtime_artifact()
        artifact["status"] = "PARTIAL"
        artifact["runtime_provenance"] = {}
        artifact["notices"] = []
        artifact["failures"] = [{
            "error_type": "MemoryError",
            "failure_classification": "oom",
            "stage": "model_load",
        }]
        self.assertEqual(
            validator.validate_runtime_artifact(artifact, require_pass=False),
            [],
        )

    def test_non_json_model_output_remains_bounded_failure_evidence(self):
        artifact = valid_runtime_artifact()
        artifact["status"] = "FAIL"
        artifact["notices"] = artifact["notices"][:1]
        artifact["notices"][0]["raw_final_answers"]["generator"] = "NOT JSON"
        artifact["notices"][0]["parsed"]["generator"] = {}
        artifact["notices"][0]["parsed"]["verifier"] = {}
        artifact["notices"][0]["validation"] = {
            "passed": False,
            "errors": ["generator stage failed closed: JSONDecodeError"],
        }
        artifact["failures"] = [{
            "error_type": "DeterministicGateFailure",
            "failure_classification": "validation",
            "stage": "final_gate",
        }]
        self.assertEqual(
            validator.validate_runtime_artifact(artifact, require_pass=False),
            [],
        )

    def test_runtime_journal_binds_exact_artifact_hash(self):
        digest = "a" * 64
        journal = {
            "schema_version": "1.0",
            "stage": "complete",
            "updated_at_utc": "2030-01-01T00:00:00+00:00",
            "details": {
                "artifact_sha256": digest,
                "completed_notices": 2,
                "error_count": 0,
                "status": "PASS",
            },
        }
        self.assertEqual(validator.validate_runtime_journal(journal, digest), [])
        self.assertTrue(validator.validate_runtime_journal(journal, "b" * 64))

    def test_prompt_budget_reserve_is_enforced(self):
        artifact = valid_runtime_artifact()
        artifact["notices"][0]["token_budgets"]["generator"] = {
            "completion_budget_tokens": 511,
            "max_length": 2048,
            "minimum_completion_tokens": 512,
            "prompt_tokens": 1537,
        }
        errors = validator.validate_runtime_artifact(artifact)
        self.assertTrue(any("violates the completion reserve" in error for error in errors))

    def test_network_import_is_rejected(self):
        errors = validator.scan_code_security("import requests\nrequests.get('example')\n")
        self.assertTrue(any("banned import" in error for error in errors))
        self.assertTrue(any("banned attribute call" in error for error in errors))

    def test_obsolete_video_tool_inventory_reference_is_absent(self):
        source = (validator.ROOT / "validate_project.py").read_text(encoding="utf-8")
        self.assertNotIn("VIDEO_TOOL_FILES", source)
        self.assertNotIn("validate_video_tools_inventory", source)


if __name__ == "__main__":
    unittest.main()
