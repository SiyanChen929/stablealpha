"""CogAlpha-Plus research framework.

The package deliberately separates candidate generation from deterministic
calculation.  Codex (or another model) may propose hypotheses and expressions;
only the versioned Python runtime is allowed to evaluate data and reveal gates.
"""

from .metrics import ICResult, summarize_ic

__all__ = ["ICResult", "summarize_ic"]
__version__ = "0.1.0"

