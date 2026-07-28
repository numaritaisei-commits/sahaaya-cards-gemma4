import io
import stat
import unittest
import zipfile

from tools import audit_wheel


def _inspect(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            info.compress_type = (
                zipfile.ZIP_STORED if name.endswith("/") else zipfile.ZIP_DEFLATED
            )
            archive.writestr(info, data)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        return audit_wheel.inspect_archive_members(archive)


def _safe_entries():
    return [
        ("keras_hub/", b"", stat.S_IFDIR | 0o755),
        ("keras_hub/__init__.py", b"VALUE = 1\n", stat.S_IFREG | 0o644),
        (
            "keras_hub-0.28.0.dist-info/METADATA",
            b"Name: keras-hub\n",
            stat.S_IFREG | 0o644,
        ),
    ]


class WheelMemberAuditTests(unittest.TestCase):
    def test_regular_allowlisted_files_and_directory_pass_member_gate(self):
        report = _inspect(_safe_entries())
        self.assertEqual(report["native_payloads"], 0)
        self.assertEqual(report["symlinks"], 0)
        self.assertEqual(report["member_count"], 3)

    def test_symlink_member_is_rejected(self):
        entries = _safe_entries()
        entries[1] = ("keras_hub/link.py", b"target", stat.S_IFLNK | 0o777)
        with self.assertRaisesRegex(ValueError, "symlink"):
            _inspect(entries)

    def test_special_file_member_is_rejected(self):
        entries = _safe_entries()
        entries[1] = ("keras_hub/pipe.py", b"", stat.S_IFIFO | 0o644)
        with self.assertRaisesRegex(ValueError, "special-file"):
            _inspect(entries)

    def test_executable_regular_file_is_rejected(self):
        entries = _safe_entries()
        entries[1] = ("keras_hub/tool.py", b"pass\n", stat.S_IFREG | 0o755)
        with self.assertRaisesRegex(ValueError, "executable"):
            _inspect(entries)

    def test_native_suffix_is_rejected(self):
        entries = _safe_entries()
        entries[1] = ("keras_hub/native.so", b"not native", stat.S_IFREG | 0o644)
        with self.assertRaisesRegex(ValueError, "native"):
            _inspect(entries)

    def test_unknown_suffix_is_rejected(self):
        entries = _safe_entries()
        entries[1] = ("keras_hub/payload.bin", b"opaque", stat.S_IFREG | 0o644)
        with self.assertRaisesRegex(ValueError, "suffix"):
            _inspect(entries)

    def test_directory_with_payload_is_rejected(self):
        entries = _safe_entries()
        entries[0] = ("keras_hub/", b"x", stat.S_IFDIR | 0o755)
        with self.assertRaisesRegex(ValueError, "directory"):
            _inspect(entries)

    def test_directory_name_with_regular_file_mode_is_rejected(self):
        entries = _safe_entries()
        entries[0] = ("keras_hub/", b"", stat.S_IFREG | 0o644)
        with self.assertRaisesRegex(ValueError, "directory"):
            _inspect(entries)

    def test_noncanonical_member_path_is_rejected(self):
        entries = _safe_entries()
        entries[1] = ("keras_hub//module.py", b"pass\n", stat.S_IFREG | 0o644)
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            _inspect(entries)


if __name__ == "__main__":
    unittest.main()
