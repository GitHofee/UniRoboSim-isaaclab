"""Clean interpreter entry point for the process-owning Isaac runtime."""

from __future__ import annotations

import sys
from multiprocessing.connection import Connection

_WORKER_PROGRESS_SCHEMA = "unirobosim-isaaclab-worker-progress/1"


def _send_progress(connection: Connection, phase: str) -> None:
    # Keep this bootstrap stdlib-only until the first progress event is sent.
    # The parent can therefore distinguish an interpreter/bootstrap stall from
    # a slow Isaac SDK launch.  The parent validates both schema and phase order.
    connection.send(
        (
            "startup_progress",
            {
                "schema": _WORKER_PROGRESS_SCHEMA,
                "phase": phase,
            },
        )
    )


def main() -> None:  # pragma: no cover - exercised by native process acceptance
    if len(sys.argv) != 2:
        raise SystemExit("worker bootstrap requires one inherited connection descriptor")
    try:
        descriptor = int(sys.argv[1])
    except ValueError as exc:
        raise SystemExit("worker connection descriptor must be an integer") from exc
    connection = Connection(descriptor)
    _send_progress(connection, "bootstrap_connected")

    # Absolute imports intentionally occur after the first progress event.  The
    # bootstrap is executed by file path with Python safe-path mode, while the
    # parent-provided PYTHONPATH anchors these imports to the exact package roots.
    from unirobosim_isaaclab.config import IsaacLabAdapterConfig

    config = connection.recv()
    if not isinstance(config, IsaacLabAdapterConfig):
        raise SystemExit("worker bootstrap received an invalid adapter configuration")
    _send_progress(connection, "config_received")

    from unirobosim_isaaclab.worker import _worker_main

    _send_progress(connection, "worker_imported")
    _worker_main(connection, config)


if __name__ == "__main__":  # pragma: no cover
    main()
