#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import builtins
import subprocess
import sys
import unittest

import evo.oauth


class TestLazyRedirectHandler(unittest.TestCase):
    """`OAuthRedirectHandler` runs a local web server, so it must not drag aiohttp into every import."""

    def test_redirect_handler_resolves_on_demand(self) -> None:
        from evo.oauth.oauth_redirect_handler import OAuthRedirectHandler

        assert evo.oauth.OAuthRedirectHandler is OAuthRedirectHandler

    def test_redirect_handler_is_exported(self) -> None:
        assert "OAuthRedirectHandler" in evo.oauth.__all__

    def test_unknown_attribute_raises(self) -> None:
        with self.assertRaises(AttributeError):
            evo.oauth.NotAThing


class TestImportWithoutAiohttp(unittest.TestCase):
    """aiohttp is an optional extra, so importing the notebook widgets must not require it."""

    def _import_with_aiohttp_blocked(self, module: str) -> subprocess.CompletedProcess:
        script = (
            "import builtins\n"
            "_real = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name.split('.')[0] == 'aiohttp':\n"
            "        raise ImportError('aiohttp is unavailable')\n"
            "    return _real(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            f"import {module}\n"
        )
        return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    def test_evo_oauth_imports_without_aiohttp(self) -> None:
        result = self._import_with_aiohttp_blocked("evo.oauth")
        assert result.returncode == 0, result.stderr

    def test_evo_notebooks_imports_without_aiohttp(self) -> None:
        result = self._import_with_aiohttp_blocked("evo.notebooks")
        assert result.returncode == 0, result.stderr

    def test_aiohttp_is_not_imported_as_a_side_effect(self) -> None:
        script = "import evo.notebooks, sys; print('aiohttp' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_guard_actually_blocks_aiohttp(self) -> None:
        """Guard against the other tests passing because the import hook is broken."""
        assert builtins.__import__ is not None
        result = self._import_with_aiohttp_blocked("evo.aio")
        assert result.returncode != 0
