"""Add-on entry module.

Avoids importing the subpackage at module import time to prevent issues
with partially-initialized packages when loaded via Blender's Extensions
namespace (bl_ext.<host>.<repo>.<id>). Instead, import lazily inside the
register/unregister functions.
"""

import importlib

def _submodule():
    return importlib.import_module(f"{__package__}.ps1d")


def register() -> None:
    _submodule().register()


def unregister() -> None:
    _submodule().unregister()
