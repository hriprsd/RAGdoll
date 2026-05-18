"""
File watcher — monitors directories for changes and triggers re-indexing.
Uses watchdog for cross-platform filesystem events.

Events are debounced (500ms) and dispatched on a background thread
so the watchdog thread is never blocked by indexing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 0.5

# File extensions to watch
WATCHED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs",
    ".java", ".rb", ".cpp", ".c", ".cs", ".swift", ".kt",
    ".md", ".mdx", ".rst", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".sql", ".sh",
}


class RagdollEventHandler(FileSystemEventHandler):
    def __init__(self, on_change: Callable[[Path, str], None]):
        """
        on_change(path, event_type) — called with 'modified', 'created', or 'deleted'.
        Calls are debounced and dispatched off the watchdog thread.
        """
        self._on_change = on_change
        self._pending: dict[str, tuple[Path, str]] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _schedule(self, path: Path, event_type: str) -> None:
        """Record a pending event and schedule a flush after the debounce window."""
        key = str(path)
        with self._lock:
            self._pending[key] = (path, event_type)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Dispatch all pending events (runs on a timer thread, not the watchdog thread)."""
        with self._lock:
            events = list(self._pending.values())
            self._pending.clear()
        for path, event_type in events:
            try:
                self._on_change(path, event_type)
            except Exception:
                logger.exception(f"Error handling {event_type} for {path}")

    def _dispatch_if_relevant(self, path: str, event_type: str) -> None:
        p = Path(path)
        if p.suffix.lower() in WATCHED_EXTENSIONS:
            self._schedule(p, event_type)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch_if_relevant(event.src_path, "modified")

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch_if_relevant(event.src_path, "created")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch_if_relevant(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._dispatch_if_relevant(event.src_path, "deleted")
            self._dispatch_if_relevant(event.dest_path, "created")


class Watcher:
    def __init__(self, paths: list[Path], on_change: Callable[[Path, str], None]):
        self._paths = paths
        self._handler = RagdollEventHandler(on_change)
        self._observer = Observer()

    def start(self) -> None:
        for path in self._paths:
            if path.exists():
                self._observer.schedule(self._handler, str(path), recursive=True)
                logger.info(f"Watching {path}")
            else:
                logger.warning(f"Watch path does not exist, skipping: {path}")
        self._observer.start()

    def stop(self) -> None:
        self._observer.stop()
        self._observer.join()
