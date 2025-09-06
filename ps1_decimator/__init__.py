"""Add-on bootstrap: wire up operators and UI panel."""

import logging
import importlib
from . import operators, ui

LOGGER = logging.getLogger(__name__)

if "bpy" in locals():
    importlib.reload(operators)
    importlib.reload(ui)


def register() -> None:
    """Register all add-on classes and UI."""
    operators.register()
    ui.register()
    print("Registered")


def unregister() -> None:
    """Unregister all add-on classes and UI."""
    ui.unregister()
    operators.unregister()
    print("Unregistered")
