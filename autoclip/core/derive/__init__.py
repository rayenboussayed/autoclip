"""Derived runtime values - the resource-aware engine, not user configuration.

Three concerns, kept separate so each has one reason to change:

* :mod:`constants` - values autoclip chooses for the user and never asks about;
* :mod:`probes` - measurements taken from the machine (pure I/O, no policy);
* :mod:`policy` - rules that turn measurements into decisions, logged as ``automation ...``.

This package replaces the old ``core/internal`` collection of ten overlapping modules. Import the
specific submodule: a re-export facade would break ``monkeypatch.setattr`` on the module that
actually reads a measurement (the patch would change the alias, not the value in use).
"""

from __future__ import annotations

from . import constants, policy, probes

__all__ = ["constants", "policy", "probes"]
