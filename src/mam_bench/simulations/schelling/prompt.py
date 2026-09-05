"""Instructions presented to Schelling Influence Actors."""

from .models import (
    ActorTurnContext,
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
