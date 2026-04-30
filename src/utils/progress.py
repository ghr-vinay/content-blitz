"""
src/utils/progress.py

Thread-local progress reporting for agent pipelines.

Usage (workflow layer):
    with progress_context(callback):
        app.invoke(...)

Usage (agent layer):
    report_progress("🔍 Researching…")

The callback is stored per-thread so it is safe with concurrent requests and
requires no changes to GraphState or agent signatures.
"""

import threading
from collections.abc import Callable

_local = threading.local()


def progress_context(callback: Callable[[str], None] | None):
    """Context manager that installs *callback* for the current thread."""

    class _Ctx:
        def __enter__(self):
            _local.callback = callback

        def __exit__(self, *_):
            _local.callback = None

    return _Ctx()


def report_progress(message: str) -> None:
    """Call the registered progress callback, if any, with *message*."""
    cb = getattr(_local, "callback", None)
    if cb is not None:
        try:
            cb(message)
        except Exception:
            pass  # never let a UI callback crash an agent
