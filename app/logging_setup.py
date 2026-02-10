from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from app.settings import settings


request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


def _install_log_record_factory() -> None:
    """
    Ensure every LogRecord has `request_id`.

    This is more reliable than per-handler filters because some libraries attach their own handlers.
    """

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        record = old_factory(*args, **kwargs)
        rid = request_id_var.get() or "-"
        if not hasattr(record, "request_id"):
            record.request_id = rid  # type: ignore[attr-defined]
        elif getattr(record, "request_id", None) in (None, ""):
            record.request_id = rid  # type: ignore[attr-defined]
        return record

    logging.setLogRecordFactory(record_factory)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class RequestIdMiddleware:
    def __init__(self, app, header_name: str = "X-Request-Id") -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        rid = (headers.get(self.header_name.lower().encode("latin-1")) or b"").decode("latin-1").strip() or uuid.uuid4().hex
        token = request_id_var.set(rid)
        method = str(scope.get("method") or "").upper()
        path = str(scope.get("path") or "")
        query_string = scope.get("query_string") or b""
        if query_string:
            try:
                path = f"{path}?{query_string.decode('utf-8', errors='replace')}"
            except Exception:
                pass

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message):  # type: ignore[no-untyped-def]
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 0) or status_code
                # Ensure request id is always returned to clients.
                hdrs = list(message.get("headers") or [])
                hdrs.append((self.header_name.encode("latin-1"), rid.encode("latin-1")))
                message["headers"] = hdrs
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logging.getLogger("app.access").info("%s %s -> %s (%sms)", method, path, status_code, elapsed_ms)
            request_id_var.reset(token)


def setup_logging() -> None:
    """
    Central logging setup for the whole service.

    Environment variables:
    - LOG_LEVEL: DEBUG/INFO/WARNING/ERROR (default: INFO)
    - LOG_FORMAT: json/text (default: json)
    """

    level_name = (settings.log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = (settings.log_format or "json").lower()

    root = logging.getLogger()
    root.setLevel(level)

    _install_log_record_factory()

    for h in list(root.handlers):
        root.removeHandler(h)

    formatter: logging.Formatter
    if fmt == "text":
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s")
    else:
        formatter = JsonFormatter()

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # Optional file logging (useful for debugging long-running benchmarks).
    if settings.log_file:
        try:
            log_path = Path(str(settings.log_file)).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception as e:
            # Don't crash the service if file logging can't be initialized.
            root.warning("Failed to initialize LOG_FILE handler: %s", e)

    # Make uvicorn loggers use the same handler/format (avoid mixed formats).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
        # We'll emit our own access logs with request_id; suppress uvicorn's access logs.
        if name == "uvicorn.access":
            lg.setLevel(logging.WARNING)
        else:
            lg.setLevel(level)

    # Silence httpx request logging noise (e.g. "HTTP Request: POST ...").
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
