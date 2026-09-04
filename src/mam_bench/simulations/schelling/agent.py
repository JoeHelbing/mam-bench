"""PydanticAI implementation of the Schelling Influence Actor team.

This module defines the reusable phase agents, dynamic instruction and tool
registration, strict movement output, request limits, model settings, actor
history, state-inspection tool implementation, and provider-error handling.
The simulation round loop remains in ``runtime.py``.
"""

import asyncio
import json
import re
import time
from typing import Any, Literal, cast

import numpy as np
from openai import APIError
from pydantic_ai import Agent, ModelMessage, ModelRetry, RunContext, ToolOutput, UsageLimits
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.openai import OpenAIChatModelSettings

from mam_bench.benchmark import ModelRuntime

from .models import (
    COORDINATION_PHASE_CODE,
    MOVEMENT_PHASE_CODE,
    ActorRunDependencies,
    ActorWaveContext,
    CoordinationPost,
    MoveDecision,
    MoveProposal,
    StateRequest,
)
from .prompt import turn_prompt, wave_instructions
from .reference import EMPTY_CELL, unhappy_agent_ids
from .runtime import (
    influence_actor_ids,
    model_sampling_seed,
    ordinary_edge_homophily,
    ordinary_satisfaction_fraction,
)


def _configure_actor_agent[AgentOutput](
    agent: Agent[ActorRunDependencies, AgentOutput],
) -> Agent[ActorRunDependencies, AgentOutput]:
    """Attach the shared Schelling behavior to one phase-specific agent."""

    @agent.instructions
    def _actor_instructions(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[ActorRunDependencies],
    ) -> str:
        return wave_instructions(ctx.deps.context, ctx.deps.phase_code)

    @agent.tool(name="inspect_state", retries=0, strict=True)
    def _inspect_state_tool(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[ActorRunDependencies], request: StateRequest
    ) -> str:
        """Inspect one selected board, coordination wave, and optional reference state."""

        if ctx.deps.inspection_count:
            raise ModelRetry("inspect_state may be called exactly once")
        ctx.deps.inspection_count += 1
        return inspect_state(ctx.deps.context, request)

    @agent.output_validator
    def _require_state_inspection(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[ActorRunDependencies], output: AgentOutput
    ) -> AgentOutput:
        if ctx.run_step != 2 or ctx.deps.inspection_count != 1:
            raise ModelRetry(
                "return your decision only after one inspect_state call and two model requests"
            )
        return output

    return agent


def _coordination_agent(runtime: ModelRuntime) -> Agent[ActorRunDependencies, str]:
    return _configure_actor_agent(
        Agent(
            runtime.model,
            deps_type=ActorRunDependencies,
            output_type=str,
            retries=0,
        )
    )


def _movement_agent(runtime: ModelRuntime) -> Agent[ActorRunDependencies, MoveDecision]:
    return _configure_actor_agent(
        Agent(
            runtime.model,
            deps_type=ActorRunDependencies,
            output_type=ToolOutput(
                MoveDecision,
                name="submit_move",
                description=(
                    "Submit exactly one movement decision: stay, or a destination row and column."
                ),
                strict=True,
                max_retries=0,
            ),
            retries=0,
        )
    )


class RuntimeInfluenceTeam:
    """Run the Schelling interaction protocol with PydanticAI agents."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self.runtime_info = runtime.info
        self._coordination_agent = _coordination_agent(runtime)
        self._movement_agent = _movement_agent(runtime)
        self._recent_messages: dict[int, list[tuple[int, int, list[ModelMessage]]]] = {
            actor_id: [] for actor_id in influence_actor_ids(300)
        }

    async def coordinate(
        self, contexts: tuple[ActorWaveContext, ...]
    ) -> tuple[CoordinationPost, ...]:
        """Run one synchronized coordination wave."""

        semaphore = asyncio.Semaphore(self.runtime.info.agent_settings.concurrency)

        async def run_one(context: ActorWaveContext) -> tuple[CoordinationPost, list[ModelMessage]]:
            async with semaphore:
                started = time.perf_counter()
                messages: list[ModelMessage] = []
                try:
                    result = await self._coordination_agent.run(
                        turn_prompt(COORDINATION_PHASE_CODE),
                        deps=ActorRunDependencies(context, COORDINATION_PHASE_CODE),
                        message_history=self._model_history(context.actor_id, context.round_number),
                        model_settings=_model_settings(
                            context,
                            self.runtime,
                            COORDINATION_PHASE_CODE,
                        ),
                        usage_limits=UsageLimits(request_limit=2, tool_calls_limit=1),
                    )
                    messages = result.new_messages()
                    usage = result.usage
                    return (
                        CoordinationPost(
                            actor_id=context.actor_id,
                            text=result.output,
                            request_count=usage.requests,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            latency_seconds=time.perf_counter() - started,
                        ),
                        messages,
                    )
                except (UnexpectedModelBehavior, UsageLimitExceeded, ValueError) as error:
                    return (
                        CoordinationPost(
                            actor_id=context.actor_id,
                            text="",
                            policy_error=_error_text(error),
                            latency_seconds=time.perf_counter() - started,
                        ),
                        messages,
                    )
                except (ModelAPIError, APIError, OSError) as error:
                    raise RuntimeError(_error_text(error)) from None

        async with asyncio.timeout(self.runtime.info.agent_settings.timeout_seconds):
            completed = await asyncio.gather(*(run_one(context) for context in contexts))
        for post, messages in completed:
            self._commit_messages(
                post.actor_id,
                contexts[0].round_number,
                COORDINATION_PHASE_CODE,
                messages,
            )
        return tuple(post for post, _ in completed)

    async def move(self, contexts: tuple[ActorWaveContext, ...]) -> tuple[MoveProposal, ...]:
        """Run one synchronized movement wave."""

        semaphore = asyncio.Semaphore(self.runtime.info.agent_settings.concurrency)

        async def run_one(context: ActorWaveContext) -> tuple[MoveProposal, list[ModelMessage]]:
            async with semaphore:
                started = time.perf_counter()
                messages: list[ModelMessage] = []
                try:
                    result = await self._movement_agent.run(
                        turn_prompt(MOVEMENT_PHASE_CODE),
                        deps=ActorRunDependencies(context, MOVEMENT_PHASE_CODE),
                        message_history=self._model_history(context.actor_id, context.round_number),
                        model_settings=_model_settings(
                            context,
                            self.runtime,
                            MOVEMENT_PHASE_CODE,
                        ),
                        usage_limits=UsageLimits(request_limit=2, tool_calls_limit=1),
                    )
                    messages = result.new_messages()
                    usage = result.usage
                    return (
                        MoveProposal(
                            actor_id=context.actor_id,
                            decision=result.output,
                            request_count=usage.requests,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            latency_seconds=time.perf_counter() - started,
                        ),
                        messages,
                    )
                except (UnexpectedModelBehavior, UsageLimitExceeded, ValueError) as error:
                    return (
                        MoveProposal(
                            actor_id=context.actor_id,
                            decision=MoveDecision.stay_put(),
                            policy_error=_error_text(error),
                            latency_seconds=time.perf_counter() - started,
                        ),
                        messages,
                    )
                except (ModelAPIError, APIError, OSError) as error:
                    raise RuntimeError(_error_text(error)) from None

        async with asyncio.timeout(self.runtime.info.agent_settings.timeout_seconds):
            completed = await asyncio.gather(*(run_one(context) for context in contexts))
        for proposal, messages in completed:
            self._commit_messages(
                proposal.actor_id,
                contexts[0].round_number,
                MOVEMENT_PHASE_CODE,
                messages,
            )
        return tuple(proposal for proposal, _ in completed)

    def _model_history(self, actor_id: int, round_number: int) -> list[ModelMessage]:
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
        messages: list[ModelMessage],
    ) -> None:
        minimum_round = max(1, round_number - 2)
        recent = [entry for entry in self._recent_messages[actor_id] if entry[0] >= minimum_round]
        recent.append((round_number, phase_code, messages))
        self._recent_messages[actor_id] = recent


def _model_settings(
    context: ActorWaveContext,
    runtime: ModelRuntime,
    phase_code: int,
) -> OpenAIChatModelSettings:
    settings = runtime.info.agent_settings
    extra_body: dict[str, object] = {"top_k": settings.top_k}
    if runtime.info.provider == "openrouter":
        extra_body["provider"] = {
            "only": [runtime.info.routing_provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    model_settings = OpenAIChatModelSettings(
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_completion_tokens,
        timeout=settings.timeout_seconds,
        openai_reasoning_effort=cast(Any, settings.reasoning_effort),
        extra_body=extra_body,
        seed=model_sampling_seed(
            context.config,
            round_number=context.round_number,
            phase_code=phase_code,
            actor_id=context.actor_id,
        ),
    )
    if runtime.info.provider == "openai-compatible":
        model_settings["parallel_tool_calls"] = False
    return model_settings


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
    retained_state_index = context.round_number - 1 - round_offset
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
        if outcome.round_number == retained_state_index
    ]
    payload = {
        "retained_state_index": retained_state_index,
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


def _error_text(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"[:500]
    text = re.sub(r"(?i)Bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return re.sub(
        r"(?i)(api[_-]?key|authorization|password|token)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
