"""Synchronized phase timings, retained separately for every fit attempt."""

import resource
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime

import torch

from .outputs import write_json


class FitTimer:
    def __init__(self, path, device, attempt):
        self.path, self.device = path, device
        self.started = time.monotonic()
        self.data: dict = dict(
            schema=1,
            attempt=attempt,
            device=device,
            started_at=datetime.now(UTC).isoformat(),
            status="running",
            phases=[],
        )

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def save(self):
        self.data["seconds"] = time.monotonic() - self.started
        write_json(self.path, self.data)

    @contextmanager
    def phase(self, name, epoch=None, synchronize=True):
        if synchronize:
            self.synchronize()
        row: dict = dict(name=name, epoch=epoch, seconds=None, status="running")
        self.data["phases"].append(row)
        self.save()
        started = time.monotonic()
        try:
            yield
            if synchronize:
                self.synchronize()
        except BaseException as error:
            row.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
            self.data.update(status=row["status"], error=type(error).__name__)
            raise
        else:
            row["status"] = "complete"
        finally:
            row["seconds"] = time.monotonic() - started
            self.save()

    def finish(self):
        self.data.update(
            status="complete",
            finished_at=datetime.now(UTC).isoformat(),
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        )
        self.save()
