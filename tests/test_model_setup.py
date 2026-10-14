"""Serving discovery and request validation without inference calls."""

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from mam_bench.config import AgentSettings, OpenRouterModel
from mam_bench.model_setup import EndpointResponse, VLLMModelMetadata, resolve_model_setup
from support import selection

PARAMETERS = ["tools", "tool_choice", "max_tokens", "temperature", "top_p", "top_k", "reasoning"]


def endpoint(**updates: object) -> dict[str, object]:
    return {
        "tag": "provider",
        "provider_name": "Provider",
        "status": 0,
        "context_length": 128000,
        "max_completion_tokens": 32000,
        "supported_parameters": PARAMETERS,
        "supports_tool_choice": {"auto": True, "required": True},
        **updates,
    }


def router(**settings: object) -> OpenRouterModel:
    return OpenRouterModel(
        runtime="openrouter",
        model="vendor/model",
        provider="provider",
        settings=AgentSettings.model_validate(settings),
    )


class ModelSetupTests(unittest.IsolatedAsyncioTestCase):
    def metadata(self, *endpoints: dict[str, object]) -> AsyncMock:
        fetch = AsyncMock(
            return_value=EndpointResponse.model_validate(
                {"data": {"endpoints": endpoints or (endpoint(),)}}
            )
        )
        patcher = patch("mam_bench.model_setup._fetch_endpoints", fetch)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fetch

    async def test_discovers_once_without_inventing_request_defaults(self) -> None:
        fetch = self.metadata(
            endpoint(),
            endpoint(tag="provider/region", context_length=64000, max_prompt_tokens=48000),
            endpoint(tag="other", provider_name="Other", context_length=1000),
        )
        selected = router()
        result = await resolve_model_setup(selected)
        fetch.assert_awaited_once_with("vendor/model")
        self.assertEqual(result.context_window_tokens, 48000)
        self.assertEqual(result.metadata["context_length"], 64000)
        self.assertEqual(result.selection.settings, selected.settings)
        self.assertIsNone(result.selection.settings.max_completion_tokens)
        self.assertIsNone(result.selection.settings.summary_completion_tokens)
        self.assertIsNone(result.selection.settings.temperature)
        self.assertIsNone(result.selection.settings.reasoning_effort)
        self.assertEqual(result.selection.settings.compaction_tail(48000), 8400)
        self.assertNotIn("context_window_tokens", result.selection.settings.model_dump())
        json.dumps(result.metadata)

    async def test_only_explicit_unsupported_parameters_fail(self) -> None:
        self.metadata(endpoint(supported_parameters=["tools", "tool_choice"]))
        await resolve_model_setup(router())
        for settings in (
            {"temperature": 1.0},
            {"top_p": 0.95},
            {"top_k": 20},
            {"reasoning_effort": "medium"},
            {"reasoning_effort": "none"},
            {"max_completion_tokens": 1000},
            {"summary_completion_tokens": 1000},
        ):
            with (
                self.subTest(settings=settings),
                self.assertRaisesRegex(ValueError, "unsupported|completion"),
            ):
                await resolve_model_setup(router(**settings))

    async def test_missing_context_and_metadata_failure_have_no_yaml_fallback(self) -> None:
        self.metadata(endpoint(context_length=None))
        with self.assertRaisesRegex(ValueError, "context"):
            await resolve_model_setup(router())
        error = ValidationError.from_exception_data("malformed metadata", [])
        with (
            patch("mam_bench.model_setup._fetch_endpoints", AsyncMock(side_effect=error)),
            self.assertRaises(ValidationError),
        ):
            await resolve_model_setup(router())

    async def test_known_limits_validate_requests_without_changing_them(self) -> None:
        self.metadata(endpoint(context_length=16000, max_completion_tokens=2000))
        for settings in (
            {"max_completion_tokens": 3000},
            {"summary_completion_tokens": 3000},
            {"compaction_tail_tokens": 12000},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                await resolve_model_setup(router(**settings))
        selected = router(max_completion_tokens=2000, temperature=0.5)
        result = await resolve_model_setup(selected)
        self.assertEqual(result.selection.settings, selected.settings)

    async def test_prompt_ceiling_is_distinct_from_generation_ceiling(self) -> None:
        self.metadata(endpoint(max_prompt_tokens=4000, max_completion_tokens=16000))
        result = await resolve_model_setup(router(max_completion_tokens=8000))
        self.assertEqual(result.context_window_tokens, 4000)
        self.assertEqual(result.selection.settings.max_completion_tokens, 8000)

    async def test_wrong_provider_unavailable_and_known_missing_tools_fail(self) -> None:
        for candidate in (
            endpoint(tag="other", provider_name="Other"),
            endpoint(status=-1),
            endpoint(supported_parameters=["max_tokens"]),
            endpoint(supported_parameters=["tools", "max_tokens"]),
        ):
            with self.subTest(endpoint=candidate):
                self.metadata(candidate)
                with self.assertRaises(ValueError):
                    await resolve_model_setup(router())

    async def test_unknown_capabilities_do_not_prevent_explicit_requests(self) -> None:
        self.metadata(endpoint(supported_parameters=None, max_completion_tokens=None))
        selected = router(temperature=0.5, max_completion_tokens=1000)
        result = await resolve_model_setup(selected)
        self.assertEqual(result.selection.settings, selected.settings)
        self.assertIsNone(result.metadata["supported_parameters"])
        self.assertFalse(result.metadata["parameter_metadata_complete"])
        # An unknown region does not erase a known incompatibility in another.
        self.metadata(endpoint(), endpoint(tag="provider/region", supported_parameters=None))
        await resolve_model_setup(selected)
        self.metadata(
            endpoint(supported_parameters=["tools", "tool_choice"]),
            endpoint(tag="provider/region", supported_parameters=None),
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            await resolve_model_setup(selected)

    async def test_completion_parameter_is_resolved_only_when_requested(self) -> None:
        self.metadata(
            endpoint(supported_parameters=["tools", "tool_choice", "max_completion_tokens"])
        )
        requested = await resolve_model_setup(router(max_completion_tokens=1000))
        assert isinstance(requested.selection, OpenRouterModel)
        self.assertEqual(requested.selection.completion_token_parameter, "max_completion_tokens")
        defaults = await resolve_model_setup(router())
        self.assertIsNone(defaults.selection.settings.max_completion_tokens)

    async def test_tool_variants_are_facts_not_rewritten_yaml(self) -> None:
        self.metadata(endpoint(supports_tool_choice={"auto": True, "required": False}))
        defaults = await resolve_model_setup(router())
        assert isinstance(defaults.selection, OpenRouterModel)
        self.assertIsNone(defaults.selection.settings.tool_choice)
        self.assertFalse(defaults.selection.required_tool_choice_supported)
        await resolve_model_setup(router(tool_choice="auto"))
        with self.assertRaisesRegex(ValueError, "tool_choice"):
            await resolve_model_setup(router(tool_choice="required"))
        self.metadata(endpoint(supports_tool_choice={"auto": False, "required": True}))
        with self.assertRaisesRegex(ValueError, "tool_choice"):
            await resolve_model_setup(router(tool_choice="auto"))
        self.metadata(endpoint(supports_tool_choice=None))
        unknown = await resolve_model_setup(router(tool_choice="required"))
        assert isinstance(unknown.selection, OpenRouterModel)
        self.assertIsNone(unknown.selection.required_tool_choice_supported)
        self.metadata(endpoint(supports_tool_choice={"auto": False, "required": False}))
        with self.assertRaisesRegex(ValueError, "tool_choice"):
            await resolve_model_setup(router())

    async def test_display_provider_normalizes_to_routing_slug(self) -> None:
        self.metadata(endpoint(tag="provider-slug", provider_name="Provider Display"))
        selected = router().model_copy(update={"provider": "Provider Display"})
        result = await resolve_model_setup(selected)
        assert isinstance(result.selection, OpenRouterModel)
        self.assertEqual(result.selection.provider, "provider-slug")

    async def test_vllm_uses_served_context_and_preserves_request_overrides(self) -> None:
        selected = selection().model_copy(
            update={"settings": AgentSettings(max_completion_tokens=2000)}
        )
        fetch = AsyncMock(return_value=VLLMModelMetadata(id="scripted", max_model_len=16000))
        with patch("mam_bench.model_setup._fetch_vllm_model", fetch):
            result = await resolve_model_setup(selected)
            self.assertIs(result.selection, selected)
            self.assertEqual(result.context_window_tokens, 16000)
            self.assertEqual(result.metadata["source"], "vllm_models")
            fetch.assert_awaited_once_with(selected)
            with self.assertRaises(ValueError):
                await resolve_model_setup(
                    selected.model_copy(
                        update={"settings": AgentSettings(max_completion_tokens=20000)}
                    )
                )
        with (
            patch(
                "mam_bench.model_setup._fetch_vllm_model",
                AsyncMock(return_value=VLLMModelMetadata(id="scripted")),
            ),
            self.assertRaisesRegex(ValueError, "context"),
        ):
            await resolve_model_setup(selected)
        for bad in (0, -1, True, "16000"):
            with self.subTest(value=bad), self.assertRaises(ValidationError):
                VLLMModelMetadata.model_validate({"id": "scripted", "max_model_len": bad})

    async def test_vllm_discovery_uses_models_route_and_checks_selected_identity(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "scripted",
                            "object": "model",
                            "created": 0,
                            "owned_by": "vllm",
                            "max_model_len": 65536,
                        }
                    ],
                },
            )

        selected = selection()
        with patch.dict(os.environ, {selected.api_key_env: "offline-test-key"}):
            for name in ("scripted", "not-served"):
                async with AsyncOpenAI(
                    api_key="offline-test-key",
                    base_url=selected.base_url,
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
                ) as client:
                    with patch("mam_bench.model_setup.AsyncOpenAI", return_value=client):
                        if name == "scripted":
                            result = await resolve_model_setup(selected)
                            self.assertEqual(result.context_window_tokens, 65536)
                            json.dumps(result.metadata)
                        else:
                            with self.assertRaisesRegex(ValueError, "not served"):
                                await resolve_model_setup(
                                    selected.model_copy(update={"model": name})
                                )
        self.assertEqual(
            [(r.method, r.url.path) for r in requests],
            [
                ("GET", "/v1/models"),
                ("GET", "/v1/models"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
