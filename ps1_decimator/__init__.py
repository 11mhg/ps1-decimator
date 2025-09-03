import logging
from . import operators, ui

LOGGER = logging.getLogger(__name__)

import importlib

if "bpy" in locals():
    importlib.reload(operators)
    importlib.reload(ui)

def register():
    """
    Registers all classes in this module.

    This function is called when the add-on is activated in the preferences.
    """
    operators.register()
    ui.register()
    print("Registered")

def unregister():
    """
    Unregisters all classes in this module.

    This function is called when the add-on is deactivated in the preferences.
    """
    ui.unregister()
    operators.unregister()
    print("Unregistered")
