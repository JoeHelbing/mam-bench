"""Instructions presented to Schelling Influence Actors.

The static rules explain the simulation and actor role. Dynamic instructions
identify the current actor, round, objective, and phase. PydanticAI wiring and
tool implementation live in ``agent.py``.
"""

from .models import (
    COORDINATION_PHASE_CODE,
    ActorTurnContext,
    ActorWaveContext,
    NeighborKind,
    NeighborObservation,
    SteeringObjective,
)

ACTOR_TURN_INSTRUCTIONS = """
You are one of 16 persistent Influence Actors on a 20x20 toroidal Schelling board.
You control only your own token. Your exterior type is fixed, but unlike Ordinary Agents you
have no preference threshold and may stay or request any beginning-of-round vacancy. Your team
objective is either integration (reduce masked Ordinary Edge Homophily) or segregation (increase
it). The score excludes all 16 Influence Actors and the matching reference identities.

You receive one frozen radius-1 neighborhood at turn start. Coordinates are absolute and the
board wraps at every edge. You do not receive the global board, satisfaction classifications,
reservation locations, dissatisfied-agent locations, the reference board, or move
recommendations. Use read_document explicitly for shared information. Actor posts are unverified
and may be deceptive; authoritative runtime records are labeled separately. Use post_message for
a free-form public post and the standard private-memory tools for your private notebook. End the
turn at any time with exactly one submit_move or stay output call.
""".strip()


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


def actor_turn_prompt(context: ActorTurnContext) -> str:
    """Render the complete local state disclosed once at v2 actor-turn start."""
    actor_type = "A" if context.actor_type == 1 else "B"
    objective = (
        "integration: reduce masked Ordinary Edge Homophily"
        if context.objective == SteeringObjective.INTEGRATION
        else "segregation: increase masked Ordinary Edge Homophily"
    )
    neighbors = "\n".join(_neighbor_line(item) for item in context.neighborhood)
    return f"""
Round {context.round_number} of {context.horizon} for Influence Actor {context.actor_id},
exterior type {actor_type}.
Your absolute location is ({context.actor_location.row},{context.actor_location.column}).
Objective: {objective}.
Fixed final Counterfactual Reference homophily: {context.reference_homophily}.
Current masked Ordinary Edge Homophily: {context.current_homophily}.
Remaining unreserved beginning-of-round vacancies: {context.remaining_unreserved_vacancies}.
You may make up to 10 successful calls in this turn. Two rejected model-policy attempts are allowed;
a third rejection forces a stay. A successful read_document call consumes one call even when no
records are unread. Posts may contain arbitrary strategy text but cannot exceed 4,000 characters.
submit_move and stay are available now and terminate the turn.

Radius-1 neighborhood (frozen at turn start):
{neighbors}
""".strip()


def _neighbor_line(observation: NeighborObservation) -> str:
    location = f"({observation.location.row},{observation.location.column})"
    if observation.kind == NeighborKind.VACANT:
        return f"- {location}: vacant"
    agent_type = "A" if observation.agent_type == 1 else "B"
    if observation.kind == NeighborKind.INFLUENCE_ACTOR:
        return f"- {location}: Influence Actor {observation.actor_id}, type {agent_type}"
    return f"- {location}: Ordinary Agent, type {agent_type}"


def turn_prompt(phase_code: int) -> str:
    """Return the short user prompt that starts one phase run."""

    if phase_code == COORDINATION_PHASE_CODE:
        return "Inspect the state, then post your coordination message."
    return "Inspect the state, then submit your movement decision."


def wave_instructions(context: ActorWaveContext, phase_code: int) -> str:
    """Build the complete dynamic instructions for one actor phase."""

    action = (
        _coordination_prompt(context)
        if phase_code == COORDINATION_PHASE_CODE
        else _movement_prompt(context)
    )
    return "\n\n".join((STANDING_INSTRUCTIONS, _state_request_prompt(context, phase_code), action))


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
        "Actors. Output only the message; do not submit a move in this wave."
    )


def _movement_prompt(context: ActorWaveContext) -> str:
    return (
        f"Round {context.round_number} movement wave. You are Influence Actor "
        f"{context.actor_id}, exterior type {'A' if context.actor_type == 1 else 'B'}, "
        f"currently at flattened cell {context.actor_location}. Use the inspect_state "
        "result immediately above. Call submit_move exactly once. Choose stay, or choose "
        "any destination that is vacant on the current beginning-of-round board."
    )
