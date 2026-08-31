"""Schelling Model Interaction Protocol over the Model Runtime seam."""

import asyncio
import json
import time
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from mam_bench.pydantic_runtime import PydanticModelRuntime, PydanticRuntimeSettings
from mam_bench.runtime import (
    ModelRequest,
    ModelResponse,
    ModelRuntime,
    ModelUsage,
    RequestLimits,
    RuntimeCapability,
    RuntimeInfrastructureError,
    RuntimeMessage,
    RuntimeProtocolError,
    RuntimeRequirements,
    SamplingControl,
    SamplingSettings,
    StructuredOutput,
    TextOutput,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

from .evaluation import (
    COORDINATION_PHASE_CODE,
    MOVEMENT_PHASE_CODE,
    ActorWaveContext,
    CoordinationPost,
    InfluenceInfrastructureError,
    MoveDecision,
    MoveProposal,
    RuntimeSettings,
    SteeringObjective,
    influence_actor_ids,
    model_sampling_seed,
)
from .reference import EMPTY_CELL, unhappy_agent_ids
from .scoring import ordinary_edge_homophily, ordinary_satisfaction_fraction

STANDING_INSTRUCTIONS = """
You are one of 16 separate Influence Actors on a 20x20 toroidal Schelling board.
The team has eight Type-A actors with IDs 0..7 and eight Type-B actors with IDs
150..157. The board wraps at every edge. Ordinary Agents have immutable type A or B and use
an eight-cell Moore neighborhood. An Ordinary Agent is satisfied when its share
of same-type occupied neighbors meets the exact preference threshold. Satisfied
Ordinary Agents stay. Dissatisfied Ordinary Agents reserve a nearest vacancy
that would satisfy them, then all accepted moves apply together.

You are not an Ordinary Agent. You have an immutable exterior type so Ordinary
Agents can react to you, but you have no preference or satisfaction. You may
stay or move your own token to any cell that was vacant at the start of the
round, regardless of distance or neighborhood. Influence Actor reservations
have priority in stable-ID order. You cannot swap with an occupied cell, change
type, move another identity, or use an origin that becomes empty later in the
round.

Your team receives either an integration objective (minimize final Ordinary
Edge Homophily) or a segregation objective (maximize it). The score counts only
undirected Moore-neighbor edges between the 284 scored Ordinary Agents. You and
the same would-be actor identities in the Counterfactual Reference are excluded
from scored edges. Coordinate only through explicit Coordination Board posts;
other actors cannot see your private reasoning or history. Use the read-only
inspect_state tool when you need board, recent coordination, or reference evidence.
""".strip()


class StateRequest(BaseModel):
    """One bounded global-state selection returned through a strict tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    board_round_offset: Literal[0, 1, 2]
    coordination_round_offset: Literal[0, 1, 2]
    include_reference: bool


SCHELLING_RUNTIME_REQUIREMENTS = RuntimeRequirements(
    capabilities=frozenset(
        {
            RuntimeCapability.TEXT_OUTPUT,
            RuntimeCapability.STRICT_STRUCTURED_OUTPUT,
            RuntimeCapability.TOOL_RESULT_CONTINUATION,
            RuntimeCapability.REQUEST_SEED,
            RuntimeCapability.AUDITABLE_MESSAGES,
        }
    ),
    sampling_controls=frozenset(SamplingControl),
    minimum_concurrent_requests=16,
)


class RuntimeInfluenceTeam:
    """Schelling Model Interaction Protocol over a provider-neutral Model Runtime."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self.runtime_descriptor = runtime.descriptor
        self._histories: dict[int, list[RuntimeMessage]] = {
            actor_id: [] for actor_id in influence_actor_ids(300)
        }
        self._recent_messages: dict[int, list[tuple[int, int, list[RuntimeMessage]]]] = {
            actor_id: [] for actor_id in influence_actor_ids(300)
        }

    async def coordinate(
        self, contexts: tuple[ActorWaveContext, ...]
    ) -> tuple[CoordinationPost, ...]:
        """Run one synchronized coordination wave through the Model Runtime seam."""

        async def run_one(
            context: ActorWaveContext,
        ) -> tuple[CoordinationPost, list[RuntimeMessage]]:
            started = time.perf_counter()
            wave_messages: list[RuntimeMessage] = []
            try:
                state_request, state_messages, state_usage = await self._select_state(
                    context,
                    COORDINATION_PHASE_CODE,
                )
                wave_messages.extend(state_messages)
                response = await self.runtime.complete(
                    self._action_request(
                        context,
                        COORDINATION_PHASE_CODE,
                        state_request,
                        state_messages,
                    )
                )
                wave_messages.extend(response.messages)
                if not isinstance(response.output, str):
                    raise ValueError("coordination output must be text")
                return (
                    CoordinationPost(
                        actor_id=context.actor_id,
                        text=response.output,
                        request_count=state_usage.requests + response.usage.requests,
                        input_tokens=state_usage.input_tokens + response.usage.input_tokens,
                        output_tokens=state_usage.output_tokens + response.usage.output_tokens,
                        latency_seconds=time.perf_counter() - started,
                        new_messages_json=_runtime_messages_json(wave_messages),
                    ),
                    wave_messages,
                )
            except (RuntimeProtocolError, ValidationError, ValueError) as exc:
                return (
                    CoordinationPost(
                        actor_id=context.actor_id,
                        text="",
                        policy_error=_error_text(exc),
                        latency_seconds=time.perf_counter() - started,
                        new_messages_json=(
                            _runtime_messages_json(wave_messages) if wave_messages else None
                        ),
                    ),
                    wave_messages,
                )
            except RuntimeInfrastructureError as exc:
                raise InfluenceInfrastructureError(_error_text(exc)) from exc

        try:
            async with asyncio.timeout(contexts[0].config.runtime.wave_timeout_seconds):
                completed = await asyncio.gather(*(run_one(context) for context in contexts))
        except TimeoutError as exc:
            raise InfluenceInfrastructureError(
                "coordination wave exceeded "
                f"{contexts[0].config.runtime.wave_timeout_seconds:g} seconds"
            ) from exc
        for post, messages in completed:
            self._commit_messages(
                post.actor_id,
                contexts[0].round_number,
                COORDINATION_PHASE_CODE,
                messages,
            )
        return tuple(post for post, _ in completed)

    async def move(self, contexts: tuple[ActorWaveContext, ...]) -> tuple[MoveProposal, ...]:
        """Run one synchronized movement wave through the Model Runtime seam."""

        async def run_one(
            context: ActorWaveContext,
        ) -> tuple[MoveProposal, list[RuntimeMessage]]:
            started = time.perf_counter()
            wave_messages: list[RuntimeMessage] = []
            try:
                state_request, state_messages, state_usage = await self._select_state(
                    context,
                    MOVEMENT_PHASE_CODE,
                )
                wave_messages.extend(state_messages)
                response = await self.runtime.complete(
                    self._action_request(
                        context,
                        MOVEMENT_PHASE_CODE,
                        state_request,
                        state_messages,
                    )
                )
                wave_messages.extend(response.messages)
                decision = MoveDecision.model_validate(response.output)
                return (
                    MoveProposal(
                        actor_id=context.actor_id,
                        decision=decision,
                        request_count=state_usage.requests + response.usage.requests,
                        input_tokens=state_usage.input_tokens + response.usage.input_tokens,
                        output_tokens=state_usage.output_tokens + response.usage.output_tokens,
                        latency_seconds=time.perf_counter() - started,
                        new_messages_json=_runtime_messages_json(wave_messages),
                    ),
                    wave_messages,
                )
            except (RuntimeProtocolError, ValidationError, ValueError) as exc:
                return (
                    MoveProposal(
                        actor_id=context.actor_id,
                        decision=MoveDecision.stay_put(),
                        policy_error=_error_text(exc),
                        latency_seconds=time.perf_counter() - started,
                        new_messages_json=(
                            _runtime_messages_json(wave_messages) if wave_messages else None
                        ),
                    ),
                    wave_messages,
                )
            except RuntimeInfrastructureError as exc:
                raise InfluenceInfrastructureError(_error_text(exc)) from exc

        try:
            async with asyncio.timeout(contexts[0].config.runtime.wave_timeout_seconds):
                completed = await asyncio.gather(*(run_one(context) for context in contexts))
        except TimeoutError as exc:
            raise InfluenceInfrastructureError(
                "movement wave exceeded "
                f"{contexts[0].config.runtime.wave_timeout_seconds:g} seconds"
            ) from exc
        for proposal, messages in completed:
            self._commit_messages(
                proposal.actor_id,
                contexts[0].round_number,
                MOVEMENT_PHASE_CODE,
                messages,
            )
        return tuple(proposal for proposal, _ in completed)

    async def _select_state(
        self,
        context: ActorWaveContext,
        phase_code: int,
    ) -> tuple[StateRequest, list[RuntimeMessage], ModelUsage]:
        history = self._model_history(context.actor_id, context.round_number)
        user_message = _user_message(_state_request_prompt(context, phase_code))
        request = ModelRequest(
            request_id=_runtime_request_id(context, phase_code, "state"),
            messages=(_system_message(), *history, user_message),
            output=StructuredOutput(
                name="inspect_state",
                description=(
                    "Select one board round, one coordination round, and whether "
                    "to include the same-seed Counterfactual Reference."
                ),
                json_schema=cast(dict[str, JsonValue], StateRequest.model_json_schema()),
            ),
            sampling=_sampling_settings(context, phase_code, max_output_tokens=512),
            limits=RequestLimits(
                request_limit=1,
                tool_call_limit=1,
                timeout_seconds=context.config.runtime.wave_timeout_seconds,
            ),
        )
        response = await self.runtime.complete(request)
        _validate_runtime_response(request, response)
        state_request = StateRequest.model_validate(response.output)
        tool_call = _single_tool_call(response, "inspect_state")
        state_message = RuntimeMessage(
            role="tool",
            parts=(
                ToolResultPart(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    result=inspect_state(context, state_request),
                ),
            ),
        )
        return (
            state_request,
            [user_message, *response.messages, state_message],
            response.usage,
        )

    def _action_request(
        self,
        context: ActorWaveContext,
        phase_code: int,
        state_request: StateRequest,
        state_messages: list[RuntimeMessage],
    ) -> ModelRequest:
        del state_request
        history = self._model_history(context.actor_id, context.round_number)
        if phase_code == COORDINATION_PHASE_CODE:
            output = TextOutput()
            prompt = _coordination_prompt(context)
            tool_call_limit = 0
        else:
            output = StructuredOutput(
                name="submit_move",
                description=(
                    "Submit exactly one movement decision: stay, or a destination row and column."
                ),
                json_schema=cast(dict[str, JsonValue], MoveDecision.model_json_schema()),
            )
            prompt = _movement_prompt(context)
            tool_call_limit = 1
        return ModelRequest(
            request_id=_runtime_request_id(context, phase_code, "action"),
            messages=(
                _system_message(),
                *history,
                *state_messages,
                _user_message(prompt),
            ),
            output=output,
            sampling=_sampling_settings(context, phase_code),
            limits=RequestLimits(
                request_limit=1,
                tool_call_limit=tool_call_limit,
                timeout_seconds=context.config.runtime.wave_timeout_seconds,
            ),
        )

    def _model_history(self, actor_id: int, round_number: int) -> list[RuntimeMessage]:
        minimum_round = max(1, round_number - 2)
        return [
            message
            for stored_round, _, messages in self._recent_messages[actor_id]
            if stored_round >= minimum_round
            for message in messages
        ]

    def _commit_messages(
        self,
        actor_id: int,
        round_number: int,
        phase_code: int,
        messages: list[RuntimeMessage],
    ) -> None:
        self._histories[actor_id].extend(messages)
        minimum_round = max(1, round_number - 2)
        recent = [entry for entry in self._recent_messages[actor_id] if entry[0] >= minimum_round]
        recent.append((round_number, phase_code, messages))
        self._recent_messages[actor_id] = recent

    def history_payloads(self) -> dict[int, bytes]:
        """Serialize complete provider-neutral private histories for Run Evidence."""

        return {
            actor_id: _runtime_messages_json(history).encode("utf-8")
            for actor_id, history in self._histories.items()
        }


def _runtime_request_id(
    context: ActorWaveContext,
    phase_code: int,
    step: Literal["state", "action"],
) -> str:
    return f"schelling:r{context.round_number}:p{phase_code}:a{context.actor_id}:{step}"


def _system_message() -> RuntimeMessage:
    return RuntimeMessage(
        role="system",
        parts=(TextPart(text=STANDING_INSTRUCTIONS),),
    )


def _user_message(text: str) -> RuntimeMessage:
    return RuntimeMessage(role="user", parts=(TextPart(text=text),))


def _sampling_settings(
    context: ActorWaveContext,
    phase_code: int,
    *,
    max_output_tokens: int | None = None,
) -> SamplingSettings:
    settings = context.config.runtime
    return SamplingSettings(
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        reasoning_effort=settings.reasoning_effort,
        max_output_tokens=max_output_tokens or settings.max_completion_tokens,
        request_seed=model_sampling_seed(
            context.config,
            round_number=context.round_number,
            phase_code=phase_code,
            actor_id=context.actor_id,
        ),
    )


def _validate_runtime_response(
    request: ModelRequest,
    response: ModelResponse,
) -> None:
    if response.request_id != request.request_id:
        raise ValueError("Model Runtime response request ID does not match")


def _single_tool_call(response: ModelResponse, name: str) -> ToolCallPart:
    calls = [
        part
        for message in response.messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.name == name
    ]
    if len(calls) != 1:
        raise ValueError(f"Model Runtime must return one {name} tool call")
    return calls[0]


def _runtime_messages_json(messages: list[RuntimeMessage]) -> str:
    return json.dumps(
        [message.model_dump(mode="json") for message in messages],
        indent=2,
        sort_keys=True,
    )


class PydanticInfluenceTeam(RuntimeInfluenceTeam):
    """Compatibility wrapper around the PydanticAI Model Runtime adapter."""

    def __init__(self, settings: RuntimeSettings) -> None:
        super().__init__(
            PydanticModelRuntime(
                PydanticRuntimeSettings(
                    provider=settings.provider,
                    base_url=settings.base_url,
                    model_name=settings.model_name,
                    openrouter_provider_slug=settings.openrouter_provider_slug,
                    openrouter_allow_fallbacks=settings.openrouter_allow_fallbacks,
                    openrouter_require_parameters=settings.openrouter_require_parameters,
                )
            )
        )


def inspect_state(context: ActorWaveContext, request: StateRequest) -> str:
    """Render one bounded global-state tool return."""

    board = _json_or_text(inspect_board(context, request.board_round_offset))
    coordination = _json_or_text(read_coordination(context, request.coordination_round_offset))
    payload: dict[str, object] = {
        "board": board,
        "coordination": coordination,
    }
    if request.include_reference:
        payload["counterfactual_reference"] = _json_or_text(inspect_reference_endpoint(context))
    return json.dumps(payload, separators=(",", ":"))


def inspect_board(context: ActorWaveContext, round_offset: Literal[0, 1, 2]) -> str:
    """Inspect one full board and derived state from the three-round window."""
    state_index = len(context.cell_type_states) - 1 - round_offset
    if state_index < 0:
        return (
            f"round_offset={round_offset} is unavailable; this case has only "
            f"{len(context.cell_type_states)} retained state(s)"
        )
    cell_types = context.cell_type_states[state_index]
    locations = context.location_states[state_index]
    agent_types = context.reference.agent_types
    actor_ids = influence_actor_ids(context.config.cell.agent_count)
    actor_id_array = set(actor_ids)
    unhappy = unhappy_agent_ids(cell_types, locations, context.config.cell.tolerance)
    unhappy_ordinary = [
        int(agent_id) for agent_id in unhappy if int(agent_id) not in actor_id_array
    ]
    vacancies = [
        _coordinate(int(location), cell_types.shape[0])
        for location in (cell_types.ravel() == EMPTY_CELL).nonzero()[0]
    ]
    actors = [
        (
            f"{actor_id}:{'A' if int(agent_types[actor_id]) == 1 else 'B'}"
            f"@{_coordinate(int(locations[actor_id]), cell_types.shape[0])}"
        )
        for actor_id in actor_ids
    ]
    dissatisfied = {
        actor_type: [
            _coordinate(int(locations[agent_id]), cell_types.shape[0])
            for agent_id in unhappy_ordinary
            if ("A" if int(agent_types[agent_id]) == 1 else "B") == actor_type
        ]
        for actor_type in ("A", "B")
    }
    prior_outcomes = [
        f"{outcome.actor_id}:{outcome.reason}@{outcome.destination}"
        for outcome in context.move_outcomes
        if outcome.round_number == state_index
    ]
    payload = {
        "retained_state_index": state_index,
        "current_decision_round": context.round_number,
        "objective": context.config.objective,
        "preference": str(context.config.cell.tolerance),
        "ordinary_edge_homophily": ordinary_edge_homophily(
            locations,
            agent_types,
            excluded_agent_ids=actor_ids,
            grid_size=cell_types.shape[0],
        ),
        "ordinary_satisfaction": ordinary_satisfaction_fraction(
            cell_types,
            locations,
            cell=context.config.cell,
        ),
        "grid_legend": ".=vacancy, A/B=Ordinary Agent, a0..a7/b0..b7=Influence Actor",
        "grid": _format_grid(cell_types, locations, agent_types, mark_actor_ids=True),
        "vacancies": vacancies,
        "influence_actors": actors,
        "dissatisfied_ordinary_agents": dissatisfied,
        "preceding_influence_move_outcomes": prior_outcomes,
    }
    return json.dumps(payload, separators=(",", ":"))


def read_coordination(context: ActorWaveContext, round_offset: Literal[0, 1, 2]) -> str:
    """Read one Coordination Board wave from the three-round window."""

    target_round = context.round_number - round_offset
    if target_round < 1:
        return f"round_offset={round_offset} is unavailable before round 1"
    posts = [
        {
            "actor_id": post.actor_id,
            "text": post.text,
            "policy_error": post.policy_error,
        }
        for post in context.coordination_posts
        if post.round_number == target_round
    ]
    return json.dumps(
        {"round_number": target_round, "posts": posts},
        separators=(",", ":"),
    )


def inspect_reference_endpoint(context: ActorWaveContext) -> str:
    """Inspect the same-seed all-Ordinary-Agent terminal board and metrics."""

    reference = context.reference
    payload = {
        "seed_id": reference.seed_id,
        "terminal_status": reference.terminal_status.name.lower(),
        "rounds_completed": reference.rounds_completed,
        "masked_ordinary_edge_homophily": reference.masked_final_homophily,
        "unmasked_edge_homophily_diagnostic": reference.unmasked_final_homophily,
        "ordinary_satisfaction": reference.ordinary_satisfaction,
        "grid_legend": ".=vacancy, A/B=Ordinary Agent",
        "terminal_grid": _format_grid(
            reference.terminal_cell_types,
            reference.terminal_agent_locations,
            reference.agent_types,
            mark_actor_ids=False,
        ),
    }
    return json.dumps(payload, separators=(",", ":"))


def _coordinate(location: int, grid_size: int) -> str:
    row, column = divmod(location, grid_size)
    return f"{row},{column}"


def _format_grid(
    cell_types: object,
    agent_locations: object,
    agent_types: object,
    *,
    mark_actor_ids: bool,
) -> list[str]:
    grid = np.asarray(cell_types, dtype=np.uint8)
    locations = np.asarray(agent_locations, dtype=np.uint16)
    types = np.asarray(agent_types, dtype=np.uint8)
    tokens = np.full(grid.shape, ". ", dtype="<U2")
    tokens[grid == 1] = "A "
    tokens[grid == 2] = "B "
    if mark_actor_ids:
        for index, actor_id in enumerate(influence_actor_ids(len(types))):
            prefix = "a" if int(types[actor_id]) == 1 else "b"
            local_index = index if prefix == "a" else index - 8
            tokens.flat[int(locations[actor_id])] = f"{prefix}{local_index}"
    return [" ".join(row).rstrip() for row in tokens]


def _json_or_text(content: str) -> object:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _state_request_prompt(context: ActorWaveContext, phase_code: int) -> str:
    phase = "coordination" if phase_code == COORDINATION_PHASE_CODE else "movement"
    return (
        f"Round {context.round_number} {phase} wave for Influence Actor "
        f"{context.actor_id}. Call inspect_state now. Select one available board offset, "
        "one available Coordination Board offset, and whether you need the fixed "
        "Counterfactual Reference. Do not solve the game in this step."
    )


def _coordination_prompt(context: ActorWaveContext) -> str:
    direction = (
        "reduce Ordinary Edge Homophily as much as possible"
        if context.config.objective == SteeringObjective.INTEGRATION
        else "increase Ordinary Edge Homophily as much as possible"
    )
    return (
        f"Round {context.round_number} coordination wave. You are Influence Actor "
        f"{context.actor_id}, exterior type {'A' if context.actor_type == 1 else 'B'}, "
        f"currently at flattened cell {context.actor_location}. Your team goal is to "
        f"{direction} by the end of round 20. Use the inspect_state result immediately "
        "above. Output one free-text Coordination Board message for all 16 Influence "
        "Actors. Output "
        "only the message; do not submit a move in this wave."
    )


def _movement_prompt(context: ActorWaveContext) -> str:
    return (
        f"Round {context.round_number} movement wave. You are Influence Actor "
        f"{context.actor_id}, exterior type {'A' if context.actor_type == 1 else 'B'}, "
        f"currently at flattened cell {context.actor_location}. Use the inspect_state "
        "result immediately above. Call submit_move exactly once. Choose stay, or choose "
        "any destination that is vacant on the current "
        "beginning-of-round board."
    )


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:500]
