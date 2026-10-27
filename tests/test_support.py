"""The test fixtures themselves: tests must never reach Modal with real credentials."""

import os
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

# The guard applies on import, as it does for every test module.
import support  # noqa: F401  # pyright: ignore[reportUnusedImport]
from mam_bench.sandboxes import ModalSettings


class OfflineModalTests(unittest.TestCase):
    def test_real_dotenv_modal_credentials_never_reach_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            Path(".env").write_text("MODAL_TOKEN_ID=real-id\nMODAL_TOKEN_SECRET=real-secret\n")
            settings = ModalSettings()  # pyright: ignore[reportCallIssue]
            self.assertEqual(settings.token_id.get_secret_value(), "offline-id")
            self.assertEqual(settings.token_secret.get_secret_value(), "offline-secret")
            # A test that clears the environment fails closed instead of reading `.env`.
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValidationError):
                ModalSettings()  # pyright: ignore[reportCallIssue]
