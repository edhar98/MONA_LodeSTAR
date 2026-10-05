"""Offline tests: never initialize a real ELab client or make network requests."""
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.elab.diagnostics import inspect_experiment as diagnostic


class ElabDiagnosticTests(unittest.TestCase):
    def run_mocked(self, response=None, error=None):
        client, api = Mock(), Mock()
        api.get_experiment.return_value = response
        api.get_experiment.side_effect = error
        factory = Mock(return_value=client)
        api_factory = Mock(return_value=api)
        modules = {
            "tools.elab.cli.elab_cli_simple": SimpleNamespace(build_api_client=factory),
            "elabapi_python": SimpleNamespace(ExperimentsApi=api_factory),
        }
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"ELAB_API_KEY": "secret-token"}), contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = diagnostic.main(["--experiment-id", "354"])
        factory.assert_called_once_with()
        api_factory.assert_called_once_with(client)
        self.assertEqual(api.mock_calls, [unittest.mock.call.get_experiment(id=354)])
        client.close.assert_called_once_with()
        return code, output.getvalue(), errors.getvalue()

    def test_only_allowlisted_scalar_metadata(self):
        code, output, errors = self.run_mocked(SimpleNamespace(id=354, title="Title secret-token", category={"body": "private"}, team=1, body="private body", access_key="private access"))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output), {"id": 354, "title": "Title [redacted]", "category": None, "team": 1})
        self.assertEqual(errors, "")
        self.assertNotIn("private", output)
        self.assertNotIn("secret-token", output)

    def test_failure_is_nonzero_without_exception_details(self):
        code, output, errors = self.run_mocked(error=RuntimeError("secret-token private body https://private-host"))
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("Unable to inspect", errors)
        self.assertNotIn("private", errors)
        self.assertNotIn("secret-token", errors)

    def test_missing_and_invalid_ids_never_access_api(self):
        for args in ([], ["--experiment-id", "0"], ["--experiment-id", "-1"], ["--experiment-id", "abc"]):
            with self.subTest(args=args), patch.object(diagnostic, "inspect_experiment") as fetch, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    diagnostic.main(args)
                self.assertEqual(error.exception.code, 2)
                fetch.assert_not_called()

    def test_help_without_credentials_or_sdk(self):
        script = Path(__file__).resolve().parents[2] / "tools/elab/diagnostics/inspect_experiment.py"
        env = {key: value for key, value in os.environ.items() if not key.startswith("ELAB_")}
        result = subprocess.run([sys.executable, "-S", str(script), "--help"], env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--experiment-id", result.stdout)
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
