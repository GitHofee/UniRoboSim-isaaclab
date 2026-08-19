"""Clean interpreter entry point for the process-owning Isaac runtime."""

from __future__ import annotations

import sys
from multiprocessing.connection import Connection

from .config import IsaacLabAdapterConfig
from .worker import _worker_main


def main() -> None:  # pragma: no cover - exercised by native process acceptance
    if len(sys.argv) != 2:
        raise SystemExit("worker bootstrap requires one inherited connection descriptor")
    try:
        descriptor = int(sys.argv[1])
    except ValueError as exc:
        raise SystemExit("worker connection descriptor must be an integer") from exc
    connection = Connection(descriptor)
    config = connection.recv()
    if not isinstance(config, IsaacLabAdapterConfig):
        raise SystemExit("worker bootstrap received an invalid adapter configuration")
    _worker_main(connection, config)


if __name__ == "__main__":  # pragma: no cover
    main()
