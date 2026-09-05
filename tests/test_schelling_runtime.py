import os
import unittest
from unittest.mock import patch

from pydantic_ai.models.openai import OpenAIChatModel

from mam_bench.benchmark import create_runtime
from mam_bench.config import OpenAICompatibleModel, OpenRouterModel


class RuntimeConstructionTests(unittest.TestCase):
    def test_builds_both_pydantic_ai_provider_paths(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test",
                "OPENROUTER_BASE_URL": "https://router.example.test/v1",
                "LOCAL_API_KEY": "test",
            },
        ):
            openrouter = create_runtime(
                OpenRouterModel(
                    id="remote",
                    runtime="openrouter",
                    model="vendor/model",
                    provider="provider-a",
                )
            )
            local = create_runtime(
                OpenAICompatibleModel(
                    id="local",
                    runtime="openai-compatible",
                    model="local/model",
                    base_url="http://127.0.0.1:8000/v1",
                    api_key_env="LOCAL_API_KEY",
                )
            )

        self.assertIsInstance(openrouter.model, OpenAIChatModel)
        self.assertEqual(openrouter.info.endpoint, "https://router.example.test/v1")
        self.assertEqual(openrouter.info.routing_provider, "provider-a")
        self.assertIsInstance(local.model, OpenAIChatModel)
        self.assertEqual(local.info.endpoint, "http://127.0.0.1:8000/v1")


if __name__ == "__main__":
    unittest.main()
