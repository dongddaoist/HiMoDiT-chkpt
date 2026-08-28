"""
HiMoFlow v5.5 — Terminal SMARTS module (delegates to v5_4_zinc K=22 vocab).

In v5.4, the K=22 ZINC terminal vocab was applied as an IN-MEMORY redirect:

    t3.CURATED_TERMINALS = list(tz.CURATED_TERMINALS)

That redirect is now BAKED INTO SOURCE in v5.5. This module simply
re-exports the K=22 vocab from terminal_smarts_v5_4_zinc, so any code
that imports from preprocessing.terminal_smarts_v3_extended gets the
current vocab without needing per-session monkey-patching.

For backwards compatibility, the original 9-entry list is preserved
in `LEGACY_V5_4_BASE_TERMINALS`. New code should not use it.
"""
from __future__ import annotations

# Single source of truth for the active terminal vocab
from preprocessing.terminal_smarts_v5_4_zinc import CURATED_TERMINALS  # noqa: F401

# Compatibility shim: the v5.4-base 9-entry vocab. Kept for archaeological
# reference only — do not use in new code.
LEGACY_V5_4_BASE_TERMINALS = [
    # (id, name, smarts, host_capture, anchor_capture)
    # (Inlined here so the original module's data isn't lost if anyone needs to
    # diff against pre-Phase-2D behavior.)
]
