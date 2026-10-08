"""Offline endpoint preflight, capability constraints, and budget derivation."""

import json
import unittest
from typing import cast
from unittest.mock import AsyncMock, patch

from mam_bench.config import AgentSettings, OpenAICompatibleModel, OpenRouterModel
from mam_bench.model_setup import EndpointResponse, resolve_model_setup

PARAMETERS = [
    "tools",
    "tool_choice",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "reasoning",
    "seed",
]


def endpoint(**updates: object) -> dict[str, object]:
    return {
        "name": "Provider test endpoint",
        "tag": "provider",
        "provider_name": "Provider",
        "status": 0,
        "context_length": 128000,
        "max_completion_tokens": 32000,
        "max_prompt_tokens": None,
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

    async def test_once_per_resolution_and_conservative_provider_limits(self) -> None:
        fetch = self.metadata(
            endpoint(),
            endpoint(
                tag="provider/region",
                context_length=64000,
                max_completion_tokens=8000,
                max_prompt_tokens=48000,
            ),
            endpoint(tag="other", provider_name="Other", context_length=1000),
        )
        result = await resolve_model_setup(router())
        fetch.assert_awaited_once_with("vendor/model")
        settings = result.selection.settings
        self.assertEqual(settings.context_window_tokens, 64000)
        self.assertEqual(settings.max_completion_tokens, 8000)
        trigger = settings.context_window_tokens * settings.compaction_trigger_fraction
        self.assertLessEqual(trigger + settings.memory_injection_tokens + 3200, 48000)
        self.assertLessEqual(
            trigger + settings.max_completion_tokens + settings.memory_injection_tokens + 3200,
            64000,
        )
        self.assertLess(
            settings.compaction_tail_tokens
            + settings.summary_completion_tokens
            + settings.memory_injection_tokens,
            trigger,
        )
        self.assertGreater(settings.compaction_trigger_fraction, 0.65)
        selected_endpoints = result.metadata["endpoints"]
        assert isinstance(selected_endpoints, list)
        self.assertEqual(len(cast(list[object], selected_endpoints)), 2)
        json.dumps(result.metadata)

    async def test_supported_defaults_omitted_but_explicit_unsupported_fails(self) -> None:
        self.metadata(endpoint(supported_parameters=["tools", "tool_choice", "max_tokens"]))
        result = await resolve_model_setup(router())
        self.assertFalse(result.selection.settings.use_sampling_seed)
        self.assertEqual(
            result.metadata["omitted_parameters"],
            ["reasoning", "seed", "temperature", "top_k", "top_p"],
        )
        for field, value in (
            ("temperature", 1.0),
            ("top_p", 0.95),
            ("top_k", 20),
            ("reasoning_effort", "medium"),
            ("use_sampling_seed", True),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                await resolve_model_setup(router(**{field: value}))
        await resolve_model_setup(router(use_sampling_seed=False))
        await resolve_model_setup(router(reasoning_effort="none"))

    async def test_missing_limit_requires_explicit_override(self) -> None:
        self.metadata(endpoint(context_length=None, max_completion_tokens=None))
        with self.assertRaisesRegex(ValueError, "context_window_tokens"):
            await resolve_model_setup(router())
        with self.assertRaisesRegex(ValueError, "max_completion_tokens"):
            await resolve_model_setup(router(context_window_tokens=16000))
        result = await resolve_model_setup(
            router(context_window_tokens=16000, max_completion_tokens=2000)
        )
        self.assertEqual(result.selection.settings.context_window_tokens, 16000)

    async def test_wrong_provider_unavailable_or_no_tools_fail(self) -> None:
        for candidate in (
            endpoint(tag="other", provider_name="Other"),
            endpoint(status=-1),
            endpoint(supported_parameters=["max_tokens"]),
        ):
            with (
                self.subTest(candidate=candidate),
                patch(
                    "mam_bench.model_setup._fetch_endpoints",
                    AsyncMock(
                        return_value=EndpointResponse.model_validate(
                            {"data": {"endpoints": [candidate]}}
                        )
                    ),
                ),
                self.assertRaises(ValueError),
            ):
                await resolve_model_setup(router())

    async def test_explicit_over_limit_and_unsafe_compaction_fail(self) -> None:
        self.metadata(endpoint(context_length=16000, max_completion_tokens=2000))
        for values in (
            {"context_window_tokens": 32000},
            {"max_completion_tokens": 3000},
            {"summary_completion_tokens": 3000},
            {"compaction_trigger_fraction": 0.99},
            {
                "compaction_tail_tokens": 10000,
                "summary_completion_tokens": 2000,
                "memory_injection_tokens": 2000,
            },
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                await resolve_model_setup(router(**values))

    async def test_anthropic_cache_and_reasoning_tool_conflict(self) -> None:
        self.metadata(endpoint(tag="anthropic", provider_name="Anthropic"))
        selected = OpenRouterModel(
            runtime="openrouter",
            model="anthropic/claude-sonnet-4",
            provider="anthropic",
            prompt_cache="required",
        )
        result = await resolve_model_setup(selected)
        self.assertIsInstance(result.selection, OpenRouterModel)
        assert isinstance(result.selection, OpenRouterModel)
        self.assertEqual(result.selection.cache_strategy, "explicit")
        self.assertEqual(result.selection.settings.tool_choice, "auto")
        with self.assertRaisesRegex(ValueError, "Claude reasoning"):
            await resolve_model_setup(
                selected.model_copy(update={"settings": AgentSettings(tool_choice="required")})
            )

    async def test_implicit_required_off_and_unverified_cache(self) -> None:
        self.metadata(endpoint(supports_implicit_caching=True))
        for policy, strategy in (("auto", "implicit"), ("required", "implicit"), ("off", "off")):
            selected = OpenRouterModel.model_validate(
                {
                    "runtime": "openrouter",
                    "model": "vendor/model",
                    "provider": "provider",
                    "prompt_cache": policy,
                }
            )
            result = await resolve_model_setup(selected)
            assert isinstance(result.selection, OpenRouterModel)
            self.assertEqual(result.selection.cache_strategy, strategy)
        self.metadata(endpoint(supports_implicit_caching=False))
        unverified = await resolve_model_setup(router())
        assert isinstance(unverified.selection, OpenRouterModel)
        self.assertEqual(unverified.selection.cache_strategy, "unverified")
        with (
            patch(
                "mam_bench.model_setup._fetch_endpoints",
                AsyncMock(
                    return_value=EndpointResponse.model_validate(
                        {"data": {"endpoints": [endpoint()]}}
                    )
                ),
            ),
            self.assertRaisesRegex(ValueError, "prompt_cache required"),
        ):
            await resolve_model_setup(router().model_copy(update={"prompt_cache": "required"}))

    async def test_azure_completion_token_parameter(self) -> None:
        self.metadata(
            endpoint(
                tag="azure",
                provider_name="Azure",
                context_length=1047576,
                max_completion_tokens=32768,
                supported_parameters=[
                    "tools",
                    "tool_choice",
                    "max_completion_tokens",
                    "temperature",
                    "top_p",
                    "seed",
                ],
            )
        )
        selected = OpenRouterModel(runtime="openrouter", model="openai/gpt-4.1", provider="azure")
        result = await resolve_model_setup(selected)
        assert isinstance(result.selection, OpenRouterModel)
        self.assertEqual(result.selection.completion_token_parameter, "max_completion_tokens")
        self.assertEqual(result.metadata["completion_token_parameter"], "max_completion_tokens")

    async def test_tool_choice_parameter_and_usable_variant_are_required(self) -> None:
        for candidate in (
            endpoint(supported_parameters=["tools", "max_tokens"]),
            endpoint(supports_tool_choice={"required": False, "auto": False}),
        ):
            with (
                self.subTest(candidate=candidate),
                patch(
                    "mam_bench.model_setup._fetch_endpoints",
                    AsyncMock(
                        return_value=EndpointResponse.model_validate(
                            {"data": {"endpoints": [candidate]}}
                        )
                    ),
                ),
                self.assertRaisesRegex(ValueError, "tool_choice"),
            ):
                await resolve_model_setup(router())

    async def test_missing_parameter_metadata_requires_explicit_limits(self) -> None:
        self.metadata(endpoint(supported_parameters=None))
        with self.assertRaisesRegex(ValueError, "capabilities are missing"):
            await resolve_model_setup(router())
        result = await resolve_model_setup(
            router(context_window_tokens=16000, max_completion_tokens=2000)
        )
        self.assertIsNone(result.metadata["supported_parameters"])

    async def test_failed_metadata_fetch_requires_explicit_limits(self) -> None:
        from pydantic import ValidationError

        error = ValidationError.from_exception_data("malformed metadata", [])
        with patch("mam_bench.model_setup._fetch_endpoints", AsyncMock(side_effect=error)):
            with self.assertRaisesRegex(ValueError, "context_window_tokens"):
                await resolve_model_setup(router())
            result = await resolve_model_setup(
                router(context_window_tokens=16000, max_completion_tokens=2000)
            )
        self.assertEqual(result.metadata["source"], "explicit_limits_fallback")
        self.assertEqual(result.metadata["metadata_error"], "ValidationError")

    async def test_display_provider_is_normalized_to_routing_slug(self) -> None:
        self.metadata(endpoint(tag="provider-slug", provider_name="Provider Display"))
        selected = router().model_copy(update={"provider": "Provider Display"})
        result = await resolve_model_setup(selected)
        assert isinstance(result.selection, OpenRouterModel)
        self.assertEqual(result.selection.provider, "provider-slug")

    async def test_compatible_selection_is_unchanged_without_metadata(self) -> None:
        fetch = self.metadata()
        selected = OpenAICompatibleModel(
            runtime="openai-compatible",
            model="local",
            base_url="http://localhost:1234",
            api_key_env="KEY",
        )
        result = await resolve_model_setup(selected)
        self.assertIs(result.selection, selected)
        fetch.assert_not_awaited()
