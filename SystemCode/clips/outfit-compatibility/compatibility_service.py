"""Compatibility import facade; the public module consumes an explicit dynamic payload."""

from travel_agent.outfits.service import run as run_outfits


def run(payload=None):
    """Return ModuleResult; no fixed result.json or implicit sample fallback."""
    return run_outfits(payload or {})
