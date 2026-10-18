"""Collect temporary model/runtime workarounds and their removal conditions.

Keep normal provider construction in runtime.py. Changes here should identify
the broken upstream behavior, retain its evidence, and say when to remove them.
No simulation rules or output-schema relaxation belongs here.

Serving plugin sources are stored here as standalone Python source strings:
vLLM's Python 3.12 environment cannot import this Python 3.14 package. Export
them with ``python -m mam_bench.kludge --export-serving DIRECTORY``. Exported
files and frozen campaign copies are artifacts, never separately maintained
implementations. Their parser names and deployment filenames stay unchanged.
"""

import ast
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic_ai import RunContext
from pydantic_ai._json_schema import JsonSchema
from pydantic_ai.messages import FinishReason, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIStreamedResponse,
)
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.settings import ModelSettings

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import VLLMModel
from mam_bench.diagnostics import ExecutionFailure


class OpenAIFixedArrayTransformer(OpenAIJsonSchemaTransformer):
    """Encode fixed homogeneous tuples using OpenAI's accepted array subset.

    Pydantic's prefixItems is valid JSON Schema, but OpenAI rejects the move
    coordinate tool with 'array schema missing items'. Equal positional schemas
    and an exact length have an equivalent items encoding. Keep heterogeneous
    tuples and all other schemas native; validation and simulation types stay
    unchanged. Evidence: large-four-model-20261016/luna6-addition/diagnostic-http
    and schema-verification.json under .scratch/model-runs/.
    Remove when the native transformer or endpoint supports this tuple form.
    """

    def transform(self, schema: JsonSchema) -> JsonSchema:
        value = schema.get("prefixItems")
        prefix = cast(list[object], value) if isinstance(value, list) else []
        if (
            schema.get("type") == "array"
            and prefix
            and schema.get("minItems") == schema.get("maxItems") == len(prefix)
            and all(item == prefix[0] for item in prefix)
            and "items" not in schema
        ):
            schema["items"] = prefix[0]
            schema.pop("prefixItems")
        return super().transform(schema)


# Task-local: concurrent sessions share a model, but never a capture directory.
vllm_stream_writer: ContextVar[ArtifactWriter | None] = ContextVar(
    "vllm_stream_writer", default=None
)


class _VLLMStreamedResponse(OpenAIStreamedResponse):
    capture_writer: ArtifactWriter | None = None
    capture_context: RunContext[object] | None = None

    async def _validate_response(self) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in super()._validate_response():
            if self.capture_writer is not None and self.capture_context is not None:
                ctx = self.capture_context
                try:
                    self.capture_writer.append(
                        f"{ctx.run_id}.jsonl",
                        {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "conversation_id": ctx.conversation_id,
                            "run_step": ctx.run_step,
                            "chunk": chunk.model_dump(mode="json", exclude_unset=True),
                        },
                    )
                except OSError as error:
                    raise ExecutionFailure(
                        "artifact_write", "provider stream write failed"
                    ) from error
            yield chunk

    def _map_finish_reason(
        self, key: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ) -> FinishReason | None:
        return "length" if str(key) == "repetition" else super()._map_finish_reason(key)

    def _map_provider_details(self, chunk: ChatCompletionChunk) -> dict[str, str]:
        details = super()._map_provider_details(chunk) or {}
        if details.get("finish_reason") == "repetition":
            details["stop_reason"] = "repetition_detected"
        return details


class VLLMChatModel(OpenAIChatModel):
    """Accept vLLM's repetition stop while preserving its real cause.

    The native OpenAI adapter rejects finish_reason='repetition'. Translate it
    to its existing incomplete-output path, then restore the vendor cause in
    provider_details. Keep strict validation for every other finish reason.
    Evidence: docs/research-issue-log.md, R12; original fix 1716471.
    Remove when the native adapter accepts and preserves vLLM repetition stops.

    Archived runs also retain SDK chunks before native event mapping, including
    token IDs emitted while the server parser buffers text. Remove this capture
    hook when native events preserve those provider fields themselves.
    """

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return _VLLMStreamedResponse

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[object] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        writer = vllm_stream_writer.get()
        if writer is not None:
            # Merge model defaults as well as per-call settings (e.g. summary caps).
            settings = {**(self.settings or {}), **(model_settings or {})}
            settings["extra_body"] = {
                **cast(dict[str, object], settings.get("extra_body") or {}),
                "return_token_ids": True,
                "include_reasoning": True,
            }
            model_settings = cast(ModelSettings, settings)
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response:
            captured = cast(_VLLMStreamedResponse, response)
            captured.capture_writer = writer
            captured.capture_context = run_context
            yield response

    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        repetition = isinstance(response, ChatCompletion) and any(
            str(choice.finish_reason) == "repetition" for choice in response.choices
        )
        if isinstance(response, ChatCompletion) and repetition:
            response = response.model_copy(
                update={
                    "choices": [
                        choice.model_copy(update={"finish_reason": "length"})
                        if str(choice.finish_reason) == "repetition"
                        else choice
                        for choice in response.choices
                    ]
                }
            )
        result = super()._process_response(response)
        if repetition:
            result.provider_details = {
                **(result.provider_details or {}),
                "finish_reason": "repetition",
                "stop_reason": "repetition_detected",
            }
        return result


def vllm_settings(selection: VLLMModel, settings: ModelSettings) -> OpenAIChatModelSettings:
    """Forward explicit thinking settings that native profile inference drops.

    Qwen3.8-27B: preserve the chat-template switch/effort from c5e1b26 on
    codex/qwen38-thinking. Native vLLM profiles deliberately exclude this
    family's different effort ladder. Off sends enable_thinking=False and no
    effort; on sends the requested effort inside chat_template_kwargs.
    Remove this case when the native provider implements that exact mapping.

    Bolt/new model names: PydanticAI drops unified 'thinking' unless its profile
    recognizes support. Forward an explicitly configured vLLM effort through
    the native OpenAI setting instead. The Bolt AgentSessions regression and
    PGX trial reproduce the omission. Remove when native profiles support the
    selected models without dropping their configured efforts.
    """
    resolved = OpenAIChatModelSettings(**settings)
    effort = selection.settings.reasoning_effort
    if effort is None:
        return resolved
    resolved.pop("thinking", None)
    if selection.model.rsplit("/", 1)[-1].lower().startswith("qwen3.8-27b"):
        chat_settings: dict[str, object] = {"enable_thinking": effort != "none"}
        if effort != "none":
            chat_settings["reasoning_effort"] = effort
        resolved["extra_body"] = {
            **cast(dict[str, object], resolved.get("extra_body") or {}),
            "chat_template_kwargs": chat_settings,
        }
    else:
        resolved["openai_reasoning_effort"] = effort
    return resolved


# Standalone serving sources: edit only here; exports require no client dependencies.
SERVING_PLUGINS: dict[str, str] = {
    # Qwen: closed empty XML argument sections must terminate (vLLM #50989,
    # XGrammar #802). Remove after upstream #837 passes the matcher regression.
    "qwen_empty_tool_kludge.py": r'''
"""Temporary Qwen XML empty-tool grammar correction for the pinned PGX server.

Python 3.12, loaded with vLLM --tool-parser-plugin; not imported by MAM-Bench.
Evidence: vllm-project/vllm#50989 and mlc-ai/xgrammar#802, proposed fix #837.
Remove when the serving image passes the accompanying matcher regression
without this plugin. Keep strict validation, native extraction and sampling.
"""

from pydantic import BaseModel
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel
from xgrammar.structural_tag import ConstStringFormat, JSONSchemaFormat, StructuralTag, TagFormat


def correct_empty_tool_arguments(tag: StructuralTag | None) -> StructuralTag | None:
    if tag is None:
        return None
    tag = tag.model_copy(deep=True)

    def visit(node: object) -> None:
        if isinstance(node, TagFormat) and isinstance(node.content, JSONSchemaFormat):
            content = node.content
            schema = content.json_schema
            if (
                content.style == "qwen_xml"
                and isinstance(schema, dict)
                and schema.get("type") == "object"
                and schema.get("additionalProperties") is False
                and not schema.get("properties")
                and not schema.get("required")
                and not schema.get("minProperties", 0)
                and schema.keys()
                <= {
                    "type",
                    "properties",
                    "required",
                    "additionalProperties",
                    "minProperties",
                    "maxProperties",
                    "title",
                    "description",
                }
            ):
                node.content = ConstStringFormat(value="")
                return
        if isinstance(node, BaseModel):
            for name in type(node).model_fields:
                visit(getattr(node, name))
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(tag)
    return tag


@ToolParserManager.register_module("qwen3_coder_empty_tool_kludge")
class QwenEmptyToolGrammarKludge(Qwen3EngineToolParser):
    def get_structural_tag(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
        *,
        reasoning: bool = False,
        strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
    ) -> StructuralTag | None:
        tag = super().get_structural_tag(request, reasoning=reasoning, strict_level=strict_level)
        return correct_empty_tool_arguments(tag)
''',
    # Gemma: pinned native parsing stringifies nullable integer tuple coordinates.
    # Evidence: docs/research-issue-log.md R04. Remove when the native parser
    # preserves these strict integer pairs and rejects tagged strings unaided.
    "gemma4_tuplefix_plugin.py": r'''
"""Scratch-only Gemma4 fix for nullable two-integer coordinate tool fields."""

import json
import re

from vllm.parser.gemma4 import STRING_DELIM, Gemma4Parser, _gemma4_arg_converter
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.gemma4_engine_tool_parser import Gemma4EngineToolParser

# Only native, unquoted JSON integer literals qualify. Tagged strings stay strings.
_PAIR = re.compile(
    r"\[\s*(-?(?:0|[1-9][0-9]*))\s*,\s*"
    r"(-?(?:0|[1-9][0-9]*))\s*\]\Z"
)


def _native_fields(raw):
    """Split top-level Gemma args; ignore commas and field names inside strings/arrays."""
    fields = {}
    parts = []
    start = depth = i = 0
    quoted = False
    while i < len(raw):
        if raw.startswith(STRING_DELIM, i):
            quoted = not quoted
            i += len(STRING_DELIM)
            continue
        char = raw[i]
        if not quoted:
            if char in "[{":
                depth += 1
            elif char in "]}":
                if depth:
                    depth -= 1
                elif char == "}" and not raw[i + 1:].strip():
                    raw = raw[:i]
                    break
                else:
                    return {}
            elif char == "," and not depth:
                parts.append(raw[start:i])
                start = i + 1
        i += 1
    if quoted or depth:
        return {}
    parts.append(raw[start:])
    for part in parts:
        name, sep, value = part.partition(":")
        name = name.strip()
        if not sep or not name.isidentifier() or name in fields:
            return {}
        fields[name] = value.strip()
    return fields


_COORD_SCHEMA = {
    "anyOf": [
        {"maxItems": 2, "minItems": 2,
         "prefixItems": [{"type": "integer"}, {"type": "integer"}], "type": "array"},
        {"type": "null"},
    ]
}


def _nullable_integer_pair(schema):
    return schema == _COORD_SCHEMA


class TupleAwareGemma4Parser(Gemma4Parser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native_args = ""
        self._arg_converter = self._convert_args

    def _convert_args(self, raw, partial):
        # Pinned vLLM calls the converter immediately before _fix_arg_types.
        self._native_args = raw
        return _gemma4_arg_converter(raw, partial)

    def _eligible_properties(self, func_name):
        if func_name not in ("move", "act"):
            return None
        matches = [getattr(tool, "function", tool) for tool in self._tools or []
                   if getattr(getattr(tool, "function", tool), "name", None) == func_name]
        if len(matches) != 1:
            return None
        strict = getattr(matches[0], "strict", None)
        if strict is not None and strict is not True:
            return None
        params = getattr(matches[0], "parameters", None)
        if not isinstance(params, dict) or set(params) != {
            "type", "properties", "required", "additionalProperties"
        } or params["type"] != "object" or params["additionalProperties"] is not False:
            return None
        properties = params["properties"]
        required = params["required"]
        if not isinstance(properties, dict) or not isinstance(required, list):
            return None
        names = set(properties)
        if len(required) != len(names) or set(required) != names:
            return None
        coords = names - {"active"}
        if not (func_name == "move" and coords == {"destination"} and names == coords
                or func_name == "act" and (
                    names == {"active", "destination"}
                    and properties["active"] == {"type": "boolean"}
                    or names == {"destination", "target_location"}
                )):
            return None
        return (properties
                if all(_nullable_integer_pair(properties[name]) for name in coords) else None)

    def _compute_arg_delta(self, idx, raw_delta):
        # Buffer only these tools: a later coordinate must not change an emitted JSON prefix.
        if self._eligible_properties(self._tool_slots[idx].name):
            return None
        return super()._compute_arg_delta(idx, raw_delta)

    def _fix_arg_types(self, args_json, func_name):
        corrected = super()._fix_arg_types(args_json, func_name)
        properties = self._eligible_properties(func_name)
        if properties is None:
            return corrected
        native = _native_fields(self._native_args)
        if set(native) != set(properties):  # No unknown or missing fields.
            return corrected
        pairs = {}
        for name, value in native.items():
            if name == "active":
                if value not in ("true", "false"):
                    return corrected
            elif value != "null":
                pair = _PAIR.fullmatch(value)
                if pair is None:
                    return corrected
                pairs[name] = pair.groups()
        if not pairs:
            return corrected
        try:
            converted = {name: [int(number) for number in pair]
                         for name, pair in pairs.items()}
        except ValueError:  # Python's integer-size limit: preserve the original response.
            return corrected
        args = json.loads(corrected)
        if set(args) != set(properties) or any(
            args.get(name) != list(pair) for name, pair in pairs.items()
        ) or any(args.get(name) is not None for name in native
                 if name != "active" and name not in pairs):
            return corrected
        if "active" in native and (type(args.get("active")) is not bool
                                   or args["active"] != (native["active"] == "true")):
            return corrected
        args.update(converted)
        return json.dumps(args, ensure_ascii=False)


class Gemma4TupleFixToolParser(Gemma4EngineToolParser):
    _parser_engine_cls = TupleAwareGemma4Parser


ToolParserManager.register_module(name="gemma4_tuplefix", module=Gemma4TupleFixToolParser)
''',
    # Bolt: pinned vLLM resolves effort into render kwargs but its parser sees
    # the original request. Evidence: pgx-nemotron35-bolt-20261015/parser_probe.py.
    # Remove when NVIDIA honors reasoning_effort or receives resolved kwargs.
    "ultra_v3_reasoning_parser.py": r'''
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("ultra_v3")
class UltraV3ReasoningParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(self, model_output, request):
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}

        # Pinned vLLM translates reasoning_effort into render kwargs without
        # updating the request seen here. Mirror its explicit-override priority.
        thinking_disabled = chat_template_kwargs.get("enable_thinking") is False or (
            "enable_thinking" not in chat_template_kwargs
            and getattr(request, "reasoning_effort", None) == "none"
        )
        if thinking_disabled:
            """
            The chat template ends the prompt with `<think></think>`, so the model
            never emits `</think>` and there is no reasoning channel at all. The
            `deepseek_r1` parser this inherits from assumes reasoning always comes
            first, so it would return the whole answer as reasoning content with an
            empty content field. Everything generated is content here, including any
            stray `</think>` the model echoes.
            """
            return None, model_output or None

        reasoning_content, final_content = super().extract_reasoning(
            model_output, request
        )

        if (
            reasoning_content
            and reasoning_content.startswith("\n")
            and self.start_token in model_output
        ):
            """
            The chat template owns the newline in `<think>\n` and re-adds it when it
            replays the turn, so `reasoning_content` must not carry it or the next
            prompt drifts by a token. Live vLLM output never contains the start token
            (the template prefilled it), so this only fires when the parser is handed
            a full assistant span: offline replay, the chat-template verifier, or
            re-parsing training data.
            """
            reasoning_content = reasoning_content[1:]

        if (
            final_content is None
            and chat_template_kwargs.get("force_nonempty_content") is True
        ):
            """
            There are rare cases where the model will output only reasoning without an
            end-think token `</think>` (e.g. reasoning exceeds max length), which
            results in empty content returned. End users may want to unilaterally
            avoid such cases and always have a content response even if the model does
            not finish its reasoning.
            """
            reasoning_content, final_content = None, reasoning_content

        return reasoning_content, final_content
''',
}


def export_serving_plugins(directory: Path) -> None:
    """Write standalone plugins for the isolated vLLM environment; never deploy them."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in SERVING_PLUGINS.items():
        source = source.removeprefix("\n")
        ast.parse(source, filename=name, feature_version=(3, 12))
        (directory / name).write_text(
            "# Generated from src/mam_bench/kludge.py; edit that source only.\n" + source,
            encoding="utf-8",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-serving", type=Path, required=True, metavar="DIRECTORY")
    export_serving_plugins(parser.parse_args().export_serving)
