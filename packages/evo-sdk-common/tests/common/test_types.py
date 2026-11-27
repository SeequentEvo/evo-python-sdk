import os
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from evo.common._types import resolve_path


@unittest.skipUnless(os.name == "nt", "Windows extended-length paths")
class TestWindowsPaths(unittest.TestCase):
    def test_long_path_can_be_opened(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ("a" * 100) / ("b" * 100) / ("c" * 100) / "data.bin"
            resolved = resolve_path(path, check_path_length=True)
            self.assertTrue(str(resolved).startswith("\\\\?\\"))
            resolved.parent.mkdir(parents=True)
            resolved.write_bytes(b"transfer data")
            self.assertEqual(b"transfer data", resolved.read_bytes())
            self.assertEqual(resolved, resolve_path(resolved, check_path_length=True))

    def test_short_path_supports_long_sibling_temporary_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ("a" * 100)
            path /= "b" * (250 - len(str(path)))
            path /= "x"
            self.assertLess(len(str(path)), 260)
            resolved = resolve_path(path, check_path_length=True)
            self.assertTrue(str(resolved).startswith("\\\\?\\"))
            resolved.parent.mkdir(parents=True)
            with NamedTemporaryFile(dir=resolved.parent) as temporary_file:
                self.assertGreater(len(temporary_file.name), 259)
                temporary_file.write(b"transfer data")

    def test_unc_path(self) -> None:
        path = resolve_path(r"\\server\share\folder\data.bin")
        self.assertEqual(r"\\?\UNC\server\share\folder\data.bin", str(path))
        self.assertEqual(path, resolve_path(path))


class TestResolvePath(unittest.TestCase):
    def test_relative_path(self) -> None:
        self.assertEqual(resolve_path("data.bin"), resolve_path(Path("folder") / ".." / "data.bin"))
