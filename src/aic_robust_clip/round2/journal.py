"""Durable local task evidence; a PID is never a success receipt."""
import json
import os
from pathlib import Path
import signal
import time
from contextlib import contextmanager


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, root):
        self.root = Path(root)
        self.state = {"status": "running", "pid": os.getpid(), "started_at": time.time(), "exit_code": None}

    def update(self, **fields):
        self.state.update(fields, updated_at=time.time())
        with (self.root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(self.state, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_json(self.root / "status.json", self.state)


@contextmanager
def task_journal(root):
    journal = Journal(root)
    def interrupted(signum, frame):
        raise InterruptedError(f"task received signal {signum}")
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        journal.update(phase="starting")
        yield journal
    except BaseException as exc:
        journal.update(status="failed", exit_code=1, error=repr(exc), auto_retry=False)
        raise
    else:
        journal.update(status="succeeded", phase="complete", exit_code=0)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
