from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from app.settings import settings


class CursorError(ValueError):
    pass


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + pad).encode("utf-8"))


def _sign(payload: bytes) -> str:
    mac = hmac.new(settings.cursor_secret.encode("utf-8"), payload, hashlib.sha256).digest()
    return _b64url_encode(mac)


def encode_cursor(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"{_b64url_encode(payload)}.{_sign(payload)}"


def decode_cursor(token: str) -> dict[str, Any]:
    try:
        payload_b64, sig = token.split(".", 1)
        payload = _b64url_decode(payload_b64)
    except Exception as e:  # noqa: BLE001
        raise CursorError("Invalid cursor") from e
    if not hmac.compare_digest(sig, _sign(payload)):
        raise CursorError("Invalid cursor signature")
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise CursorError("Invalid cursor payload") from e


@dataclass(frozen=True)
class SeekAnchor:
    ts: datetime
    message_id: str


def parse_seek_anchor(cursor: Optional[str], *, expected_kind: str) -> Optional[SeekAnchor]:
    if not cursor:
        return None
    payload = decode_cursor(cursor)
    if payload.get("kind") != expected_kind:
        raise CursorError("Cursor kind mismatch")
    ts = datetime.fromisoformat(payload["ts"])
    return SeekAnchor(ts=ts, message_id=str(payload["message_id"]))


def make_seek_cursor(*, kind: str, ts: datetime, message_id: str, extra: Optional[dict[str, Any]] = None) -> str:
    payload: dict[str, Any] = {"kind": kind, "ts": ts.isoformat(), "message_id": message_id}
    if extra:
        payload.update(extra)
    return encode_cursor(payload)
