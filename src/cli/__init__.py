"""Utilities for exposing the CLI package API without eager imports."""

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

__all__ = ["main", "data_commands", "train_commands", "backtest_commands"]

_SUBMODULES = {
    "main": "src.cli.main",
    "data_commands": "src.cli.commands.data_commands",
    "train_commands": "src.cli.commands.train_commands",
    "backtest_commands": "src.cli.commands.backtest_commands",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - module level behaviour
    """Lazily import CLI submodules when they are first accessed."""

    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - module level behaviour
    return sorted(set(__all__) | set(globals().keys()))


if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from . import main
    from .commands import backtest_commands, data_commands, train_commands
