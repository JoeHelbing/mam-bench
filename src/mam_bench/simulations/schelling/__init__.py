"""Schelling domain classes."""

from .occupants import Agent, ModelControlledAgent, OrdinaryAgent
from .simulation import SchellingSim

__all__ = ["Agent", "ModelControlledAgent", "OrdinaryAgent", "SchellingSim"]
