"""Serving preflight through real discovery responses, without inference calls."""

import json
import logging
import os
import unittest
from unittest.mock import patch

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from mam_bench.config import AgentSettings, ModelSelection
from mam_bench.model_setup import ResolvedModelSetup, resolve_model_setup
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


def router(**settings: object) -> ModelSelection:
    return ModelSelection(
        runtime="openrouter",
        model="vendor/model",
        provider="provider",
        settings=AgentSettings.model_validate(settings),
    )


@patch.dict(
    os.environ,
    {"OPENROUTER_API_KEY": "offline-test-key", "UNUSED_OFFLINE_TEST_KEY": "offline-test-key"},
)
class ModelSetupTests(unittest.IsolatedAsyncioTestCase):
    async def discover(
        self, selected: ModelSelection, payload: object, catalog: list[object] | None = None
    ) -> ResolvedModelSetup:
        # Exercise SDK HTTP serialization and application response parsing.
        def respond(request: httpx.Request) -> httpx.Response:
            if selected.runtime == "openrouter" and request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": catalog or []})
            expected = (
                "/v1/models" if selected.runtime == "vllm" else "/v1/models/vendor/model/endpoints"
            )
            self.assertEqual((request.method, request.url.path), ("GET", expected))
            return httpx.Response(200, json=payload)

        async with AsyncOpenAI(
            api_key="offline-test-key",
            base_url="https://provider.example.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            with patch("mam_bench.model_setup.AsyncOpenAI", return_value=client):
                return await resolve_model_setup(selected)

    async def test_router_selects_regional_limits_without_changing_requested_defaults(self) -> None:
        selected = router()
        original = selected.model_dump()
        setup = await self.discover(
            selected,
            {
                "data": {
                    "endpoints": [
                        endpoint(),
                        endpoint(
                            tag="provider/region", context_length=64000, max_prompt_tokens=48000
                        ),
                        endpoint(tag="other", provider_name="Other", context_length=1000),
                    ]
                }
            },
        )
        self.assertEqual(setup.context_window_tokens, 48000)
        self.assertEqual(setup.metadata["context_length"], 64000)
        self.assertEqual(setup.selection.settings.max_completion_tokens, 16000)
        self.assertEqual(setup.selection.settings.summary_completion_tokens, 16000)
        self.assertEqual(setup.selection.settings.compaction_tail(48000), 8400)
        self.assertEqual(selected.model_dump(), original)
        self.assertIsNone(setup.selection.settings.temperature)
        self.assertIsNone(setup.selection.settings.reasoning_effort)
        self.assertIsNone(selected.settings.max_completion_tokens)
        self.assertIsNone(selected.settings.summary_completion_tokens)
        self.assertNotIn("context_window_tokens", setup.selection.settings.model_dump())
        json.dumps(setup.metadata)

    async def test_router_keeps_explicit_caps_separate_from_prompt_and_output_ceilings(
        self,
    ) -> None:
        for name, facts, settings, context, caps in (
            (
                "explicit limits",
                endpoint(context_length=16000, max_completion_tokens=2000),
                AgentSettings(
                    max_completion_tokens=2000, summary_completion_tokens=1000, temperature=0.5
                ),
                16000,
                (2000, 1000),
            ),
            (
                "output ceiling",
                endpoint(context_length=16000, max_completion_tokens=2000),
                AgentSettings(),
                16000,
                (2000, 2000),
            ),
            (
                "prompt ceiling",
                endpoint(max_prompt_tokens=4000, max_completion_tokens=16000),
                AgentSettings(max_completion_tokens=8000),
                4000,
                (8000, 16000),
            ),
        ):
            with self.subTest(scenario=name):
                selected = router().model_copy(update={"settings": settings})
                setup = await self.discover(selected, {"data": {"endpoints": [facts]}})
                self.assertEqual(setup.context_window_tokens, context)
                self.assertEqual(
                    (
                        setup.selection.settings.max_completion_tokens,
                        setup.selection.settings.summary_completion_tokens,
                    ),
                    caps,
                )
                self.assertEqual(setup.selection.settings.temperature, settings.temperature)
                self.assertEqual(selected.settings, settings)

    async def test_router_uses_known_capabilities_and_leaves_unknown_requests_unchanged(
        self,
    ) -> None:
        selected = router(
            temperature=0.5, max_completion_tokens=1000, summary_completion_tokens=2000
        )
        for facts in (
            [endpoint(supported_parameters=None, max_completion_tokens=None)],
            [endpoint(), endpoint(tag="provider/region", supported_parameters=None)],
        ):
            with self.subTest(endpoints=facts):
                setup = await self.discover(selected, {"data": {"endpoints": facts}})
                self.assertEqual(setup.selection.settings, selected.settings)
                self.assertFalse(setup.metadata["parameter_metadata_complete"])
        self.assertIsNone(
            (
                await self.discover(
                    selected, {"data": {"endpoints": [endpoint(supported_parameters=None)]}}
                )
            ).metadata["supported_parameters"]
        )
        for cap, expected in ((None, 32000), (1000, 1000)):
            with self.subTest(completion_cap=cap):
                setup = await self.discover(
                    router(max_completion_tokens=cap),
                    {
                        "data": {
                            "endpoints": [
                                endpoint(
                                    supported_parameters=[
                                        "tools",
                                        "tool_choice",
                                        "max_completion_tokens",
                                    ]
                                )
                            ]
                        }
                    },
                )
                self.assertEqual(
                    setup.selection.completion_token_parameter, "max_completion_tokens"
                )
                self.assertEqual(setup.selection.settings.max_completion_tokens, expected)
        for requested in (None, "auto"):
            with self.subTest(tool_choice=requested):
                setup = await self.discover(
                    router(tool_choice=requested),
                    {
                        "data": {
                            "endpoints": [
                                endpoint(supports_tool_choice={"auto": True, "required": False})
                            ]
                        }
                    },
                )
                self.assertEqual(setup.selection.settings.tool_choice, requested)
                self.assertFalse(setup.selection.required_tool_choice_supported)
        setup = await self.discover(
            router(tool_choice="required"),
            {"data": {"endpoints": [endpoint(supports_tool_choice=None)]}},
        )
        self.assertIsNone(setup.selection.required_tool_choice_supported)

    async def test_router_normalizes_a_display_name_for_provider_routing(self) -> None:
        selected = router().model_copy(update={"provider": "Provider Display"})
        setup = await self.discover(
            selected,
            {
                "data": {
                    "endpoints": [endpoint(tag="provider-slug", provider_name="Provider Display")]
                }
            },
        )
        self.assertEqual(setup.selection.provider, "provider-slug")
        self.assertEqual(selected.provider, "Provider Display")

    async def test_router_preflight_rejects_unusable_or_incompatible_serving_facts(self) -> None:
        scenarios = [
            ("missing context", [endpoint(context_length=None)], router(), "context"),
            (
                "wrong provider",
                [endpoint(tag="other", provider_name="Other")],
                router(),
                "active endpoints",
            ),
            ("unavailable", [endpoint(status=-1)], router(), "active endpoints"),
            ("no tools", [endpoint(supported_parameters=["max_tokens"])], router(), "tools"),
            (
                "no tool choice",
                [endpoint(supported_parameters=["tools", "max_tokens"])],
                router(),
                "tool_choice",
            ),
            (
                "action too large",
                [endpoint(context_length=16000, max_completion_tokens=2000)],
                router(max_completion_tokens=3000),
                "limit",
            ),
            (
                "summary too large",
                [endpoint(context_length=16000, max_completion_tokens=2000)],
                router(summary_completion_tokens=3000),
                "limit",
            ),
            (
                "tail too large",
                [endpoint(context_length=16000)],
                router(compaction_tail_tokens=12000),
                "compaction",
            ),
            (
                "required unsupported",
                [endpoint(supports_tool_choice={"auto": True, "required": False})],
                router(tool_choice="required"),
                "tool_choice",
            ),
            (
                "auto unsupported",
                [endpoint(supports_tool_choice={"auto": False, "required": True})],
                router(tool_choice="auto"),
                "tool_choice",
            ),
            (
                "no usable mode",
                [endpoint(supports_tool_choice={"auto": False, "required": False})],
                router(),
                "tool_choice",
            ),
        ]
        for settings in ({}, {"max_completion_tokens": 1000}, {"summary_completion_tokens": 1000}):
            scenarios.append(
                (
                    f"missing completion parameter {settings}",
                    [endpoint(supported_parameters=["tools", "tool_choice"])],
                    router(**settings),
                    "completion",
                )
            )
        for name, endpoints, selected, error in scenarios:
            with self.subTest(scenario=name), self.assertRaisesRegex(ValueError, error):
                await self.discover(selected, {"data": {"endpoints": endpoints}})
        with self.assertRaises(ValidationError):
            await self.discover(router(), {"data": {"endpoints": [{"tag": "provider"}]}})

    async def test_router_forwards_explicit_settings_and_records_advertised_efforts(self) -> None:
        # Endpoint metadata no longer vetoes values; the provider decides at request time.
        terse = endpoint(supported_parameters=["tools", "tool_choice", "max_tokens"])
        reasoning = {"supported_efforts": ["max", "high", "low", "none"], "default_effort": "high"}
        catalog: list[object] = [{"id": "vendor/model", "reasoning": reasoning}]
        for effort, warned in (("max", False), ("xhigh", True)):
            with self.subTest(effort=effort), self.assertLogs("mam_bench", "INFO") as logs:
                logging.getLogger("mam_bench").info("sentinel")
                setup = await self.discover(
                    router(temperature=1.0, top_k=20, reasoning_effort=effort),
                    {"data": {"endpoints": [terse]}},
                    catalog,
                )
                self.assertEqual(setup.selection.settings.reasoning_effort, effort)
                self.assertEqual(setup.metadata["advertised_reasoning"], reasoning)
                self.assertEqual(
                    any("reasoning_effort.unadvertised" in line for line in logs.output), warned
                )
        setup = await self.discover(router(), {"data": {"endpoints": [endpoint()]}})
        self.assertIsNone(setup.metadata["advertised_reasoning"])

    async def test_vllm_preflight_checks_identity_context_and_generation_budgets(self) -> None:
        scenarios = (
            ("defaults", "scripted", 65536, AgentSettings(), 16384, 16384, None),
            (
                "explicit caps",
                "scripted",
                16000,
                AgentSettings(max_completion_tokens=6000, summary_completion_tokens=1000),
                6000,
                1000,
                None,
            ),
            ("wrong model", "not-served", 16000, AgentSettings(), None, None, "not served"),
            ("missing context", "scripted", None, AgentSettings(), None, None, "context"),
            (
                "oversized output",
                "scripted",
                16000,
                AgentSettings(max_completion_tokens=20000),
                None,
                None,
                "limit",
            ),
        )
        for name, model_id, context, settings, action_cap, summary_cap, error in scenarios:
            with self.subTest(scenario=name):
                selected = selection().model_copy(update={"settings": settings})
                payload = {"data": [{"id": model_id, "max_model_len": context}]}
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        await self.discover(selected, payload)
                    continue
                setup = await self.discover(selected, payload)
                self.assertEqual(setup.context_window_tokens, context)
                self.assertEqual(setup.selection.settings.max_completion_tokens, action_cap)
                self.assertEqual(setup.selection.settings.summary_completion_tokens, summary_cap)
                self.assertEqual(selected.settings, settings)
                self.assertEqual(setup.metadata["source"], "vllm_models")
                json.dumps(setup.metadata)
        for bad in (0, -1, True, "16000"):
            with self.subTest(context=bad), self.assertRaises(ValidationError):
                await self.discover(
                    selection(), {"data": [{"id": "scripted", "max_model_len": bad}]}
                )
