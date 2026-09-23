"""Bounded, per-worker Kit compilation activity reader used only during startup."""
import re
from pathlib import Path

_PATTERN = re.compile(rb"\[(\d[\d,]*)ms\].*Waiting for RtPso async group async compilation: (\d+) seconds so far")

class KitStartupActivity:
    def __init__(self, path: Path):
        self.path = path
        self.offset = path.stat().st_size  # Never credit pre-existing/old content.
        self.pending = b""
        self.last_elapsed_ms = -1
        self.last_wait_seconds = -1

    def poll(self) -> bool:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read(65536)
                self.offset = stream.tell()
        except OSError:
            return False
        lines = (self.pending + data).split(b"\n")
        self.pending = lines.pop()[-4096:]
        active = False
        for line in lines:
            match = _PATTERN.search(line)
            if match is None:
                continue
            elapsed = int(match[1].replace(b",", b""))
            waited = int(match[2])
            if elapsed > self.last_elapsed_ms and waited > self.last_wait_seconds:
                self.last_elapsed_ms = elapsed
                self.last_wait_seconds = waited
                active = True
        return active
