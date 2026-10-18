"""Real-time log capture and background task monitoring manager."""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from typing import List, Optional
from PySide6.QtCore import QObject, Signal

LOG_FILE_PATH = os.path.abspath("iqa_gui_runtime.log")


class LogStreamEmitter(QObject):
    """Qt emitter for thread-safe cross-thread log streaming."""
    log_received = Signal(str)


class StreamTee(io.TextIOBase):
    """Tee stdout/stderr to both original stream, Qt emitter, and persistent log file."""

    def __init__(self, original_stream, emitter: LogStreamEmitter, prefix: str = ""):
        super().__init__()
        self.original_stream = original_stream
        self.emitter = emitter
        self.prefix = prefix
        self._lock = threading.Lock()
        self._file = None
        try:
            self._file = open(LOG_FILE_PATH, "a", encoding="utf-8", errors="replace", buffering=1)
        except Exception:
            pass

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            try:
                self.original_stream.write(s)
                self.original_stream.flush()
            except Exception:
                pass

            try:
                self.emitter.log_received.emit(s)
            except Exception:
                pass

            if self._file:
                try:
                    self._file.write(s)
                    self._file.flush()
                except Exception:
                    pass

        return len(s)

    def flush(self) -> None:
        with self._lock:
            try:
                self.original_stream.flush()
            except Exception:
                pass
            if self._file:
                try:
                    self._file.flush()
                except Exception:
                    pass


class QtLoggingHandler(logging.Handler):
    """Logging handler routing python logging records to Qt emitter."""

    def __init__(self, emitter: LogStreamEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) + "\n"
            self.emitter.log_received.emit(msg)
        except Exception:
            pass


class LogManager:
    """Singleton managing log streams, recent buffers, and background monitoring."""

    _instance: Optional[LogManager] = None

    def __init__(self):
        self.emitter = LogStreamEmitter()
        self._buffer: List[str] = []
        self._buffer_limit = 3000
        self._lock = threading.Lock()
        self._installed = False

        self.emitter.log_received.connect(self._append_to_buffer)

    @classmethod
    def get_instance(cls) -> LogManager:
        if cls._instance is None:
            cls._instance = LogManager()
        return cls._instance

    def _append_to_buffer(self, text: str) -> None:
        with self._lock:
            self._buffer.append(text)
            if len(self._buffer) > self._buffer_limit:
                self._buffer = self._buffer[-self._buffer_limit:]

    def get_all_logs(self) -> str:
        with self._lock:
            return "".join(self._buffer)

    def clear_logs(self) -> None:
        with self._lock:
            self._buffer.clear()
        try:
            with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass

    def install(self) -> None:
        """Install stdout/stderr hooks and logging handler."""
        if self._installed:
            return
        self._installed = True

        # Hook sys.stdout and sys.stderr
        sys.stdout = StreamTee(sys.stdout, self.emitter, prefix="")
        sys.stderr = StreamTee(sys.stderr, self.emitter, prefix="[ERR] ")

        # Hook standard logging
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        handler = QtLoggingHandler(self.emitter)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        logging.info("IQA-GUI 日志监视器已成功就绪，正在监听后台标准输出与错误日志...")
