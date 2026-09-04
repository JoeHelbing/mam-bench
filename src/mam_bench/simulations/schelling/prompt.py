"""Instructions presented to Schelling Influence Actors.

The static rules explain the simulation and actor role. Dynamic instructions
identify the current actor, round, objective, and phase. PydanticAI wiring and
tool implementation live in ``agent.py``.
"""

from .models import (
    COORDINATION_PHASE_CODE,
    ActorWaveContext,
    SteeringObjective,
)

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
