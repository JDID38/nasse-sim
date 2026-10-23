"""Regression tests for public artifact extraction boundaries."""
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from scripts.download_artifacts import extract_verified


class ArchiveTests(unittest.TestCase):
    def check_archive(self, name, symlink=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "test.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                entry = zipfile.ZipInfo(name)
                if symlink:
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                bundle.writestr(entry, "content")
            target = root / "output"
            target.mkdir()
            if symlink or name.startswith("../"):
                with self.assertRaises(ValueError):
                    extract_verified(archive, target)
                self.assertEqual(list(target.iterdir()), [])
            else:
                extract_verified(archive, target)
                self.assertEqual((target / name).read_text(), "content")

    def test_nested_file(self):
        self.check_archive("campaign/results.json")

    def test_parent_traversal(self):
        self.check_archive("../outside.txt")

    def test_symlink(self):
        self.check_archive("link", symlink=True)


if __name__ == "__main__":
    unittest.main()
