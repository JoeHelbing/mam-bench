"""Offline support for the Schelling reference-data lifecycle.

The modules in this package generate the complete reference sweep, validate its
artifacts, summarize its outcomes, and publish the compact resources consumed by
the benchmark. Normal benchmark execution does not import this package; it loads
the already-published evaluation fixture through ``schelling.fixture`` instead.
"""
