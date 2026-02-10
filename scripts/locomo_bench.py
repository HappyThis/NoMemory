from __future__ import annotations

import argparse
import asyncio
import atexit
import hashlib
import json
import math
import re
import random
import string
import sys
import time
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _stable_seed_int(*parts: Any) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "big", signed=False)

def _stable_id_hex(*parts: Any, n: int = 32) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:n]

def _fmt_eta(seconds: float) -> str:
    try:
        sec = float(seconds)
    except Exception:
        return "-"
    if not math.isfinite(sec):
        return "inf"
    s = int(max(0.0, sec))
    mm, ss = divmod(s, 60)
    hh, mm = divmod(mm, 60)
    if hh:
        return f"{hh}h{mm:02d}m{ss:02d}s"
    if mm:
        return f"{mm}m{ss:02d}s"
    return f"{ss}s"


def _parse_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    # LoCoMo session time format: "1:56 pm on 8 May, 2023"
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", s, re.IGNORECASE)
    if m:
        h, mm, ap, day, mon, year = m.groups()
        s2 = f"{int(h)}:{mm} {ap.upper()} on {int(day)} {mon}, {year}"
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
            try:
                dt = datetime.strptime(s2, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    # common: "2023-01-01T00:00:00Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # common: "YYYY-MM-DD HH:MM:SS"
        try:
            dt = datetime.fromisoformat(s.replace(" ", "T"))
        except ValueError:
            return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _chunk(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _sanitize_user_id(s: str) -> str:
    s2 = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", s.strip())
    return s2[:128] if len(s2) > 128 else s2


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _f1(pred: str, gold: str) -> float:
    pred_n = _normalize_answer(pred)
    gold_n = _normalize_answer(gold)
    if not pred_n and not gold_n:
        return 1.0
    if not pred_n or not gold_n:
        return 0.0
    pred_toks = pred_n.split()
    gold_toks = gold_n.split()
    common: dict[str, int] = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _max_f1(pred: str, gold: Any) -> Optional[float]:
    if gold is None:
        return None
    if isinstance(gold, str):
        return _f1(pred, gold)
    if isinstance(gold, list):
        vals = []
        for g in gold:
            if isinstance(g, str):
                vals.append(_f1(pred, g))
        return max(vals) if vals else None
    return None


@dataclass(frozen=True)
class Turn:
    ts: datetime
    role: str
    content: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class QAItem:
    qa_idx: int
    question: str
    answer: Any
    category: Any
    evidence: Any
    now: Optional[datetime]


class _Tee:
    def __init__(self, a, b) -> None:
        self._a = a
        self._b = b

    def write(self, s: str) -> int:
        na = self._a.write(s)
        self._b.write(s)
        return na

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()

    def isatty(self) -> bool:  # pragma: no cover
        try:
            return bool(self._a.isatty())
        except Exception:
            return False


def _extract_locomo_samples(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for key in ("data", "samples", "dialogs", "dialogues"):
            v = raw.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # some files may be a single sample dict
        if "conversation" in raw:
            return [raw]
    raise ValueError("Unrecognized LoCoMo JSON format (expected list or dict with data/samples)")


def _extract_turns(sample: dict[str, Any]) -> list[Turn]:
    convo = sample.get("conversation")
    if not isinstance(convo, dict):
        raise ValueError("sample.conversation missing or not an object")

    speaker_a = convo.get("speaker_a")
    speaker_b = convo.get("speaker_b")
    sample_id = sample.get("sample_id") or sample.get("dia_id") or ""

    # sessions are named like "session_1", "session_2", ...
    session_keys = [k for k in convo.keys() if isinstance(k, str) and re.fullmatch(r"session_\d+", k)]
    session_keys.sort(key=lambda x: int(x.split("_", 1)[1]))

    turns: list[Turn] = []
    for sk in session_keys:
        session = convo.get(sk)
        if not isinstance(session, list):
            continue
        dt_key = f"{sk}_date_time"
        session_dt = _parse_dt(convo.get(dt_key)) or _now_utc()
        # LoCoMo does not provide per-turn timestamps, only per-session start time.
        # Synthesize per-turn timestamps by adding a random 1–5s gap (deterministic per sample+session)
        # to keep ordering stable while being more "natural" than 1ms increments.
        rng = random.Random(_stable_seed_int(sample_id, sk))
        offset = timedelta(0)
        for idx, t in enumerate(session):
            if not isinstance(t, dict):
                continue
            speaker = t.get("speaker")
            role = None
            if isinstance(speaker, str) and isinstance(speaker_a, str) and speaker == speaker_a:
                role = "user"
            elif isinstance(speaker, str) and isinstance(speaker_b, str) and speaker == speaker_b:
                role = "assistant"
            elif isinstance(t.get("role"), str):
                role = str(t["role"])
            elif isinstance(speaker, str) and speaker.lower() in ("user", "assistant", "system"):
                role = speaker.lower()
            else:
                role = "user"

            text = t.get("text")
            if not isinstance(text, str):
                text = str(text or "")

            # If the dataset includes an image caption, bring it into text so a text-only model can use it.
            blip = t.get("blip_caption")
            if isinstance(blip, str) and blip.strip():
                text = (text.rstrip() + "\n[Image] " + blip.strip()).strip()

            meta = {
                "locomo": {
                    "sample_id": sample.get("sample_id") or sample.get("dia_id") or "",
                    "session": sk,
                    "dia_id": t.get("dia_id"),
                    "speaker": speaker,
                }
            }
            if "img_url" in t:
                meta["locomo"]["img_url"] = t.get("img_url")
            if "query" in t:
                meta["locomo"]["query"] = t.get("query")

            if idx > 0:
                offset += timedelta(seconds=float(rng.uniform(1.0, 5.0)))
            ts = session_dt + offset
            turns.append(Turn(ts=ts, role=role, content=text, meta=meta))

    # stable ordering
    turns.sort(key=lambda x: (x.ts, x.meta["locomo"].get("dia_id") or ""))
    return turns


def _extract_qa(sample: dict[str, Any], *, fallback_now: Optional[datetime]) -> list[QAItem]:
    qa = sample.get("qa") or sample.get("qas") or sample.get("questions")
    if not isinstance(qa, list):
        return []
    out: list[QAItem] = []
    for idx, item in enumerate(qa, start=1):
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        now = _parse_dt(item.get("now")) or _parse_dt(item.get("query_time")) or _parse_dt(item.get("query_date_time")) or fallback_now
        out.append(
            QAItem(
                qa_idx=idx,
                question=q.strip(),
                answer=item.get("answer"),
                category=item.get("category"),
                evidence=item.get("evidence"),
                now=now,
            )
        )
    return out


def _post_json(client: httpx.Client, url: str, *, headers: dict[str, str] | None = None, payload: dict[str, Any]) -> Any:
    r = client.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()


def _get_json(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    r = client.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()


def _wait_for_embeddings(client: httpx.Client, *, user_id: str, expected: int, timeout_sec: float) -> None:
    if expected <= 0:
        return
    t0 = time.monotonic()
    sleep_s = 0.25
    last_emb = -1
    while True:
        status = _get_json(client, f"/v1/users/{user_id}/embeddings/status")
        enabled = bool(status.get("enabled"))
        emb = int(status.get("embeddings") or 0)
        if not enabled:
            return
        if emb >= expected:
            return

        elapsed = time.monotonic() - t0
        if elapsed >= float(timeout_sec):
            raise TimeoutError(
                f"embeddings_not_ready user_id={user_id} embeddings={emb} expected>={expected} timeout_sec={timeout_sec}"
            )

        if emb != last_emb:
            last_emb = emb
            print(f"wait embeddings user_id={user_id} embeddings={emb}/{expected} elapsed={_fmt_eta(elapsed)}")
        time.sleep(sleep_s)
        sleep_s = min(2.0, sleep_s * 1.5)


async def _post_json_async(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any],
    max_retries: int,
    backoff_sec: float,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last_err = e
            status = None
            if isinstance(e, httpx.HTTPStatusError):
                status = e.response.status_code
            retryable = status in (429, 502, 503, 504) or status is None
            if not retryable or attempt >= max_retries:
                raise
            sleep_s = float(backoff_sec) * (2**attempt) * (0.5 + random.random())
            await asyncio.sleep(sleep_s)
    if last_err is not None:
        raise last_err
    raise RuntimeError("unreachable")


def _safe_load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _completed_keys_from_predictions(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    for obj in _safe_load_jsonl(path):
        sample_id = obj.get("sample_id")
        qa_idx = obj.get("qa_idx")
        if isinstance(sample_id, str) and isinstance(qa_idx, int):
            done.add((sample_id, qa_idx))
    return done


def _parse_category_1_5(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if 1 <= v <= 5 else None
    if isinstance(v, float) and v.is_integer():
        i = int(v)
        return i if 1 <= i <= 5 else None
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            i = int(s)
            return i if 1 <= i <= 5 else None
    return None


def _parse_only_category_arg(v: str | None) -> int | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    mapping = {
        "1": 1,
        "single": 1,
        "single-hop": 1,
        "singlehop": 1,
        "2": 2,
        "multi": 2,
        "multi-hop": 2,
        "multihop": 2,
        "3": 3,
        "temporal": 3,
        "time": 3,
        "4": 4,
        "commonsense": 4,
        "common-sense": 4,
        "5": 5,
        "adversarial": 5,
        "adv": 5,
    }
    if s in mapping:
        return mapping[s]
    for k, name in _CATEGORY_LABELS.items():
        if s == name.lower():
            return k
    return None


_CATEGORY_LABELS: dict[int, str] = {
    1: "Single-hop",
    2: "Multi-hop",
    3: "Temporal",
    4: "Commonsense",
    5: "Adversarial",
}


def _category_label(v: Any) -> str:
    cat = _parse_category_1_5(v)
    if cat is None:
        return str(v)
    return _CATEGORY_LABELS.get(cat, str(cat))


def _has_gold_answer(gold: Any) -> bool:
    if gold is None:
        return False
    if isinstance(gold, str):
        return bool(gold.strip())
    if isinstance(gold, list):
        for g in gold:
            if isinstance(g, str) and g.strip():
                return True
        return False
    return True


def _score_f1(*, pred_answer: str, gold_answer: Any, category: Any) -> Optional[float]:
    """
    LoCoMo F1 with a special rule for Adversarial (category=5):
      - If gold exists: score as normal.
      - If gold is missing: pred=="unknown" => 1.0 else 0.0
    """
    cat = _parse_category_1_5(category)
    if cat == 5 and not _has_gold_answer(gold_answer):
        return 1.0 if pred_answer.strip().lower() == "unknown" else 0.0
    return _max_f1(pred_answer, gold_answer)


def _f1_value_or_zero(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


def _row_f1_value(obj: dict[str, Any]) -> float:
    f1 = obj.get("f1")
    if isinstance(f1, (int, float)) and math.isfinite(float(f1)):
        return float(f1)
    pred = obj.get("pred_answer")
    if not isinstance(pred, str):
        pred = str(pred or "")
    gold = obj.get("gold_answer")
    cat = obj.get("category")
    scored = _score_f1(pred_answer=pred, gold_answer=gold, category=cat)
    return float(scored) if isinstance(scored, (int, float)) and math.isfinite(float(scored)) else 0.0


def _new_cat_sums() -> dict[int, float]:
    return {i: 0.0 for i in range(1, 6)}


def _new_cat_counts() -> dict[int, int]:
    return {i: 0 for i in range(1, 6)}


def _fmt_cat_f1_means(cat_sum: dict[int, float], cat_cnt: dict[int, int]) -> str:
    parts: list[str] = []
    for i in range(1, 6):
        cnt = int(cat_cnt.get(i) or 0)
        mean = (float(cat_sum.get(i) or 0.0) / cnt) if cnt > 0 else 0.0
        parts.append(f"{_CATEGORY_LABELS.get(i, str(i))}={mean:.3f}")
    return " ".join(parts)


def _fmt_cat_judge_means(judge_sum: dict[int, float], judge_cnt: dict[int, int]) -> str:
    parts: list[str] = []
    for i in range(1, 6):
        cnt = int(judge_cnt.get(i) or 0)
        mean = (float(judge_sum.get(i) or 0.0) / cnt) if cnt > 0 else 0.0
        parts.append(f"{_CATEGORY_LABELS.get(i, str(i))}={mean:.3f}")
    return " ".join(parts)


def _cat_f1_totals_from_predictions(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    """
    Build run-level category (1..5) F1 sum/count from predictions.jsonl.

    If a row has f1==null/missing, it is treated as 0 for aggregation.
    If there are duplicate (sample_id, qa_idx) rows, the last one wins.
    """
    by_key: dict[tuple[str, int], tuple[int, float]] = {}
    for obj in _safe_load_jsonl(path):
        sample_id = obj.get("sample_id")
        qa_idx = obj.get("qa_idx")
        if not (isinstance(sample_id, str) and isinstance(qa_idx, int)):
            continue
        cat = _parse_category_1_5(obj.get("category"))
        if cat is None:
            continue
        f1 = _row_f1_value(obj)
        by_key[(sample_id, qa_idx)] = (cat, f1)

    cat_sum = _new_cat_sums()
    cat_cnt = _new_cat_counts()
    for _, (cat, f1) in by_key.items():
        cat_sum[cat] += float(f1)
        cat_cnt[cat] += 1
    return cat_sum, cat_cnt


def _cat_judge_totals_from_predictions(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    """
    Build run-level category (1..5) judge score sum/count from predictions.jsonl.

    - Rows without judge fields are excluded.
    - If there are duplicate (sample_id, qa_idx) rows, the last one wins.
    """
    by_key: dict[tuple[str, int], tuple[int, float] | None] = {}
    for obj in _safe_load_jsonl(path):
        sample_id = obj.get("sample_id")
        qa_idx = obj.get("qa_idx")
        if not (isinstance(sample_id, str) and isinstance(qa_idx, int)):
            continue
        cat = _parse_category_1_5(obj.get("category"))
        if cat is None:
            continue
        score = obj.get("judge_score")
        if isinstance(score, bool):
            score = 1.0 if score else 0.0
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            by_key[(sample_id, qa_idx)] = (cat, float(score))
        else:
            by_key[(sample_id, qa_idx)] = None

    cat_sum = _new_cat_sums()
    cat_cnt = _new_cat_counts()
    for v in by_key.values():
        if v is None:
            continue
        cat, score = v
        cat_sum[cat] += float(score)
        cat_cnt[cat] += 1
    return cat_sum, cat_cnt


def _backfill_missing_judge(
    *,
    preds_path: Path,
    base_url: str,
    max_retries: int,
    backoff_sec: float,
    concurrency: int,
    progress_every: int = 50,
) -> int:
    """
    Rewrite predictions.jsonl in-place, adding judge_* fields for rows that don't have them.

    Requires the service to expose POST /v1/bench/locomo/judge.
    Returns the number of updated rows.
    """
    if not preds_path.exists():
        return 0
    conc = max(1, int(concurrency))

    raws = preds_path.read_text(encoding="utf-8").splitlines()
    if not raws:
        return 0

    records: list[tuple[str, Any]] = []
    for raw in raws:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            records.append(("raw", raw.rstrip("\n")))
            continue
        if not isinstance(obj, dict):
            records.append(("raw", raw.rstrip("\n")))
            continue
        records.append(("obj", obj))

    todo: list[tuple[int, dict[str, Any]]] = []
    for idx, (kind, val) in enumerate(records):
        if kind != "obj":
            continue
        obj = val
        if obj.get("judge_label") in ("CORRECT", "WRONG") and isinstance(obj.get("judge_score"), (int, float, bool)):
            continue
        q = obj.get("question")
        pred = obj.get("pred_answer")
        if not isinstance(q, str) or not q.strip():
            continue
        if not isinstance(pred, str):
            pred = str(pred or "")
        payload = {
            "question": q,
            "gold_answer": obj.get("gold_answer"),
            "pred_answer": pred,
            "category": _parse_category_1_5(obj.get("category")),
            "now": obj.get("now"),
        }
        todo.append((idx, payload))

    if not todo:
        return 0

    async def _run() -> dict[int, tuple[str, int, str | None]]:
        results: dict[int, tuple[str, int, str | None]] = {}
        q: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
        for it in todo:
            q.put_nowait(it)
        for _ in range(conc):
            q.put_nowait(None)

        fatal_evt = asyncio.Event()
        fatal_err: Exception | None = None
        updated_local = 0
        lock = asyncio.Lock()

        async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(180.0)) as client:

            async def worker() -> None:
                nonlocal fatal_err, updated_local
                while True:
                    if fatal_evt.is_set():
                        return
                    item = await q.get()
                    if item is None:
                        q.task_done()
                        return
                    rec_idx, payload = item
                    try:
                        resp = await _post_json_async(
                            client,
                            "/v1/bench/locomo/judge",
                            payload=payload,
                            max_retries=max_retries,
                            backoff_sec=backoff_sec,
                        )
                        if not isinstance(resp, dict):
                            raise RuntimeError(f"invalid judge response: {resp!r}")
                        label = resp.get("label")
                        if label not in ("CORRECT", "WRONG"):
                            raise RuntimeError(f"invalid judge label: {label!r}")
                        score = 1 if label == "CORRECT" else 0
                        reason = resp.get("reason")
                        reason_s = reason.strip() if isinstance(reason, str) and reason.strip() else None
                        async with lock:
                            results[rec_idx] = (label, score, reason_s)
                            updated_local += 1
                            if progress_every and updated_local % int(progress_every) == 0:
                                print(f"judge backfill updated={updated_local}/{len(todo)} file={preds_path}")
                    except Exception as e:
                        if not fatal_evt.is_set():
                            fatal_err = e
                            fatal_evt.set()
                        q.task_done()
                        return
                    q.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(conc)]
            join_task = asyncio.create_task(q.join())
            fatal_wait = asyncio.create_task(fatal_evt.wait())
            done, pending = await asyncio.wait({join_task, fatal_wait}, return_when=asyncio.FIRST_COMPLETED)

            if fatal_evt.is_set():
                for t in pending:
                    t.cancel()
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                raise RuntimeError(f"judge_backfill_failed updated={len(results)}/{len(todo)} err={fatal_err}") from fatal_err

            for t in pending:
                t.cancel()
            for w in workers:
                await w
        return results

    results = asyncio.run(_run()) if conc > 1 else None

    # Fallback to sequential mode if requested.
    if conc <= 1:
        results = {}
        with httpx.Client(base_url=base_url, timeout=180.0) as client:
            for k, payload in todo:
                resp_obj = _post_json(client, "/v1/bench/locomo/judge", payload=payload)
                if not isinstance(resp_obj, dict):
                    raise RuntimeError(f"invalid judge response: {resp_obj!r}")
                label = resp_obj.get("label")
                if label not in ("CORRECT", "WRONG"):
                    raise RuntimeError(f"invalid judge label: {label!r}")
                score = 1 if label == "CORRECT" else 0
                reason = resp_obj.get("reason")
                reason_s = reason.strip() if isinstance(reason, str) and reason.strip() else None
                results[k] = (label, score, reason_s)

    tmp = preds_path.with_suffix(preds_path.suffix + ".tmp")
    bak = preds_path.with_suffix(preds_path.suffix + f".bak.{_now_utc().strftime('%Y%m%dT%H%M%SZ')}")
    updated = 0
    with tmp.open("w", encoding="utf-8") as out:
        for idx, (kind, val) in enumerate(records):
            if kind == "raw":
                out.write(str(val).rstrip("\n") + "\n")
                continue
            obj = val
            r = results.get(idx)
            if r is not None:
                label, score, reason = r
                obj["judge_label"] = label
                obj["judge_score"] = score
                if reason:
                    obj["judge_reason"] = reason
                updated += 1
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    shutil.copy2(preds_path, bak)
    tmp.replace(preds_path)
    if updated:
        print(f"judge backfill done updated={updated} total_missing={len(todo)} file={preds_path} concurrency={conc}")
    return updated


def _cleanup_prediction_errors(*, preds_path: Path) -> int:
    """
    Remove rows with non-empty `error` from predictions.jsonl so they can be retried on resume.

    Dropped rows are appended to a sidecar file: predictions.errors.jsonl.
    Returns the number of dropped rows.
    """
    if not preds_path.exists():
        return 0
    errors_path = preds_path.with_name("predictions.errors.jsonl")
    kept: list[str] = []
    dropped: list[str] = []
    for raw in preds_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            dropped.append(raw)
            continue
        if not isinstance(obj, dict):
            dropped.append(raw)
            continue
        err = obj.get("error")
        if err is None or (isinstance(err, str) and not err.strip()):
            kept.append(raw)
        else:
            dropped.append(raw)

    preds_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    if dropped:
        with errors_path.open("a", encoding="utf-8") as f:
            for r in dropped:
                f.write(r.rstrip("\n") + "\n")
    return len(dropped)


def _compute_summary_from_predictions(*, preds_path: Path, run_id: str, base_url: str, samples: int) -> dict[str, Any]:
    f1_by_cat: dict[str, list[float]] = defaultdict(list)
    judge_by_cat: dict[str, list[float]] = defaultdict(list)
    total = 0
    scored = 0
    judge_scored = 0
    for obj in _safe_load_jsonl(preds_path):
        total += 1
        cat = _category_label(obj.get("category"))
        f1 = obj.get("f1")
        if isinstance(f1, (int, float)):
            scored += 1
        f1_by_cat[cat].append(_row_f1_value(obj))
        js = obj.get("judge_score")
        if isinstance(js, bool):
            js = 1.0 if js else 0.0
        if isinstance(js, (int, float)) and math.isfinite(float(js)):
            judge_scored += 1
            judge_by_cat[cat].append(float(js))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    label_order = {v: k for (k, v) in _CATEGORY_LABELS.items()}
    def _sort_key(k: str) -> tuple[int, str]:
        return (label_order.get(k, 999), k)

    return {
        "run_id": run_id,
        "base_url": base_url,
        "samples": samples,
        "qa_total": total,
        "qa_scored": scored,
        "f1_mean": _mean([x for xs in f1_by_cat.values() for x in xs]),
        "f1_by_category": {k: {"count": len(v), "mean": _mean(v)} for (k, v) in sorted(f1_by_cat.items(), key=lambda kv: _sort_key(kv[0]))},
        "judge_scored": judge_scored,
        "judge_mean": _mean([x for xs in judge_by_cat.values() for x in xs]),
        "judge_by_category": {k: {"count": len(v), "mean": _mean(v)} for (k, v) in sorted(judge_by_cat.items(), key=lambda kv: _sort_key(kv[0]))},
        "predictions_path": str(preds_path),
    }


async def _run_qa_concurrent(
    *,
    base_url: str,
    user_id: str,
    sample_id: str,
    qa_items: list[QAItem],
    preds_f,
    done_keys: set[tuple[str, int]],
    concurrency: int,
    sleep_sec: float,
    max_retries: int,
    backoff_sec: float,
    progress_sec: float,
    progress_every: int,
    cat_sum: dict[int, float],
    cat_cnt: dict[int, int],
    judge_enabled: bool,
    judge_sum: dict[int, float],
    judge_cnt: dict[int, int],
) -> None:
    write_lock = asyncio.Lock()
    tasks: list[tuple[int, QAItem]] = []
    for qa in qa_items:
        qa_idx = int(qa.qa_idx)
        if (sample_id, qa_idx) not in done_keys:
            tasks.append((qa_idx, qa))

    if not tasks:
        print(f"qa sample_id={sample_id} all_done total={len(qa_items)}")
        return

    total_to_run = len(tasks)
    completed = 0
    errors = 0
    last_idx: int | None = None
    in_flight = 0
    t0 = time.monotonic()
    fatal_evt = asyncio.Event()
    fatal_msg: str | None = None

    q: asyncio.Queue[tuple[int, QAItem] | None] = asyncio.Queue()
    for t in tasks:
        q.put_nowait(t)

    for _ in range(max(1, int(concurrency))):
        q.put_nowait(None)

    headers = {"X-User-Id": user_id}
    timeout = httpx.Timeout(180.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        stop = asyncio.Event()

        async def reporter() -> None:
            if progress_sec <= 0:
                return
            while not stop.is_set():
                await asyncio.sleep(float(progress_sec))
                elapsed = max(1e-6, time.monotonic() - t0)
                rate = completed / elapsed
                remaining = total_to_run - completed
                eta = (remaining / rate) if rate > 0 else float("inf")
                li = last_idx if last_idx is not None else "-"
                print(
                    f"qa sample_id={sample_id} done={completed}/{total_to_run} errors={errors} "
                    f"inflight={in_flight} rate={rate:.2f}/s eta={_fmt_eta(eta)} last={li} "
                    f"f1_mean_by_cat={_fmt_cat_f1_means(cat_sum, cat_cnt)}"
                    + (f" judge_mean_by_cat={_fmt_cat_judge_means(judge_sum, judge_cnt)}" if judge_enabled else "")
                )

        async def worker() -> None:
            nonlocal completed, errors, last_idx, fatal_msg, in_flight
            while True:
                if fatal_evt.is_set():
                    return
                item = await q.get()
                if item is None:
                    q.task_done()
                    return
                qa_idx, qa = item
                last_idx = qa_idx

                total_payload: dict[str, Any] = {"question": qa.question}
                if qa.now is not None:
                    total_payload["now"] = qa.now.isoformat()

                pred_answer = ""
                evidence = []
                judge_label: str | None = None
                judge_score: int | None = None
                judge_reason: str | None = None
                err = None
                in_flight += 1
                try:
                    try:
                        resp = await _post_json_async(
                            client,
                            "/v1/bench/locomo/qa",
                            headers=headers,
                            payload=total_payload,
                            max_retries=max_retries,
                            backoff_sec=backoff_sec,
                        )
                        pred_answer = str(resp.get("answer") or "")
                        evidence = resp.get("evidence") or []
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        errors += 1
                        # Stop the whole benchmark run immediately on the first error.
                        if not fatal_evt.is_set():
                            fatal_msg = f"qa_failed sample_id={sample_id} qa_idx={qa_idx} error={err}"
                            fatal_evt.set()
                        q.task_done()
                        return

                    f1 = _score_f1(pred_answer=pred_answer, gold_answer=qa.answer, category=qa.category)
                    if judge_enabled:
                        try:
                            jresp = await _post_json_async(
                                client,
                                "/v1/bench/locomo/judge",
                                headers=headers,
                                payload={
                                    "question": qa.question,
                                    "gold_answer": qa.answer,
                                    "pred_answer": pred_answer,
                                    "category": _parse_category_1_5(qa.category),
                                    "now": qa.now.isoformat() if qa.now else None,
                                },
                                max_retries=max_retries,
                                backoff_sec=backoff_sec,
                            )
                            if isinstance(jresp, dict):
                                judge_label = str(jresp.get("label") or "")
                                if judge_label in ("CORRECT", "WRONG"):
                                    judge_score = 1 if judge_label == "CORRECT" else 0
                                reason = jresp.get("reason")
                                judge_reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
                            if judge_score is None:
                                raise RuntimeError(f"invalid judge response: {jresp!r}")
                        except Exception as e:
                            err = f"JudgeError {type(e).__name__}: {e}"
                            errors += 1
                            if not fatal_evt.is_set():
                                fatal_msg = f"judge_failed sample_id={sample_id} qa_idx={qa_idx} error={err}"
                                fatal_evt.set()
                            q.task_done()
                            return
                finally:
                    in_flight -= 1
                row = {
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "qa_idx": qa_idx,
                    "category": qa.category,
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "pred_answer": pred_answer,
                    "f1": f1,
                    "judge_label": judge_label,
                    "judge_score": judge_score,
                    "judge_reason": judge_reason,
                    "now": qa.now.isoformat() if qa.now else None,
                    "evidence_count": len(evidence) if isinstance(evidence, list) else None,
                    "error": err,
                }

                async with write_lock:
                    preds_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    preds_f.flush()
                    done_keys.add((sample_id, qa_idx))
                    cat = _parse_category_1_5(qa.category)
                    if cat is not None:
                        cat_sum[cat] = float(cat_sum.get(cat) or 0.0) + _f1_value_or_zero(f1)
                        cat_cnt[cat] = int(cat_cnt.get(cat) or 0) + 1
                        if judge_enabled and judge_score is not None:
                            judge_sum[cat] = float(judge_sum.get(cat) or 0.0) + float(judge_score)
                            judge_cnt[cat] = int(judge_cnt.get(cat) or 0) + 1
                    completed += 1
                    if progress_every and completed % int(progress_every) == 0:
                        elapsed = max(1e-6, time.monotonic() - t0)
                        rate = completed / elapsed
                        remaining = total_to_run - completed
                        eta = (remaining / rate) if rate > 0 else float("inf")
                        print(
                            f"qa sample_id={sample_id} done={completed}/{total_to_run} errors={errors} "
                            f"inflight={in_flight} rate={rate:.2f}/s eta={_fmt_eta(eta)} last={qa_idx} "
                            f"f1_mean_by_cat={_fmt_cat_f1_means(cat_sum, cat_cnt)}"
                            + (f" judge_mean_by_cat={_fmt_cat_judge_means(judge_sum, judge_cnt)}" if judge_enabled else "")
                        )

                if sleep_sec and sleep_sec > 0:
                    await asyncio.sleep(float(sleep_sec))
                q.task_done()

        report_task = asyncio.create_task(reporter())
        workers = [asyncio.create_task(worker()) for _ in range(max(1, int(concurrency)))]
        join_task = asyncio.create_task(q.join())
        fatal_wait = asyncio.create_task(fatal_evt.wait())
        done, pending = await asyncio.wait({join_task, fatal_wait}, return_when=asyncio.FIRST_COMPLETED)

        if fatal_evt.is_set():
            for t in pending:
                t.cancel()
            for w in workers:
                w.cancel()
            stop.set()
            report_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if fatal_msg:
                raise RuntimeError(fatal_msg)
            raise RuntimeError("qa_failed")

        for t in pending:
            t.cancel()
        stop.set()
        await report_task
        for w in workers:
            await w

    elapsed = max(1e-6, time.monotonic() - t0)
    print(
        f"qa sample_id={sample_id} done={completed}/{total_to_run} errors={errors} elapsed={_fmt_eta(elapsed)} "
        f"f1_mean_by_cat={_fmt_cat_f1_means(cat_sum, cat_cnt)}"
        + (f" judge_mean_by_cat={_fmt_cat_judge_means(judge_sum, judge_cnt)}" if judge_enabled else "")
    )


def _download_file(*, url: str, dest: Path, timeout_sec: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with httpx.Client(timeout=timeout_sec) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    if chunk:
                        f.write(chunk)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description="LoCoMo benchmark runner: ingest + QA + F1 report")
    ap.add_argument(
        "--data",
        default=None,
        help="Path to locomo10.json. If omitted, defaults to ./data/locomo/locomo10.json (auto-download if missing).",
    )
    ap.add_argument(
        "--download-url",
        default="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        help="URL to download locomo10.json from (used with --download).",
    )
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="Overwrite existing --data file when used with --download.",
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download locomo10.json automatically; fail if --data does not exist.",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="NoMemory service base URL.")
    ap.add_argument("--outdir", default="traces/locomo", help="Output directory for results.")
    ap.add_argument("--user-prefix", default="locomo", help="User id prefix.")
    ap.add_argument("--run-id", default=None, help="Run id suffix. Default: UTC timestamp.")
    ap.add_argument("--limit-samples", type=int, default=None, help="Limit number of dialogue samples.")
    ap.add_argument("--limit-qa", type=int, default=None, help="Limit number of QA items per sample.")
    ap.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep seconds between QA calls.")
    ap.add_argument("--wait-embeddings-sec", type=float, default=0.0, help="(Optional) Extra sleep after ingest. Embeddings are now waited for automatically by default.")
    ap.add_argument("--embeddings-timeout-sec", type=float, default=120.0, help="Max seconds to wait for embeddings after ingest.")
    ap.add_argument("--no-wait-embeddings", action="store_true", help="Do not wait for embeddings after ingest.")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run (requires --run-id; appends to predictions.jsonl).")
    ap.add_argument("--concurrency", type=int, default=2, help="Number of concurrent QA requests (default: 2).")
    ap.add_argument("--max-retries", type=int, default=3, help="Max retries per QA request on retryable errors (default: 3).")
    ap.add_argument("--retry-backoff-sec", type=float, default=1.0, help="Base backoff seconds for retries (default: 1.0).")
    ap.add_argument("--skip-ingest", action="store_true", help="Skip ingest and only run QA (useful with --resume).")
    ap.add_argument("--progress-sec", type=float, default=5.0, help="Print progress every N seconds during QA (default: 5).")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N completed QA (default: 10).")
    ap.add_argument("--judge", action="store_true", help="Compute LLM-as-judge metric (requires /v1/bench/locomo/judge).")
    ap.add_argument("--judge-concurrency", type=int, default=2, help="Concurrent judge requests for backfill (default: 2).")
    ap.add_argument("--only-category", default=None, help="Run only one QA category (1-5 or name, e.g. temporal).")
    ap.add_argument("--log-file", default=None, help="Append stdout/stderr logs to this file.")
    ap.add_argument("--summarize-only", action="store_true", help="Only (re)write summary.json from predictions.jsonl; do not ingest or run QA.")
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing run's predictions.jsonl when not using --resume.")
    args = ap.parse_args()

    if args.summarize_only and not args.run_id:
        raise ValueError("--summarize-only requires --run-id")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id")
    if args.concurrency is None or int(args.concurrency) <= 0:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries is None or int(args.max_retries) < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.retry_backoff_sec is None or float(args.retry_backoff_sec) <= 0:
        raise ValueError("--retry-backoff-sec must be > 0")
    if args.judge_concurrency is None or int(args.judge_concurrency) <= 0:
        raise ValueError("--judge-concurrency must be >= 1")
    only_cat = _parse_only_category_arg(args.only_category)
    if args.only_category is not None and only_cat is None:
        raise ValueError(f"--only-category invalid: {args.only_category!r} (expected 1-5 or a name like temporal)")

    data_path = Path(args.data) if args.data else Path("data/locomo/locomo10.json")
    if not data_path.exists():
        if args.no_download:
            raise FileNotFoundError(f"--data not found: {data_path}")
        print(f"download locomo10.json url={args.download_url} dest={data_path}")
        _download_file(url=str(args.download_url), dest=data_path)
    elif args.force_download:
        if args.no_download:
            raise ValueError("--force-download conflicts with --no-download")
        print(f"download locomo10.json url={args.download_url} dest={data_path} (force)")
        _download_file(url=str(args.download_url), dest=data_path)

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    samples = _extract_locomo_samples(raw)
    if args.limit_samples is not None:
        samples = samples[: max(0, int(args.limit_samples))]

    run_id = args.run_id or _now_utc().strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    preds_path = outdir / "predictions.jsonl"
    summary_path = outdir / "summary.json"
    mapping_path = outdir / "user_mapping.json"

    if args.log_file:
        log_path = Path(str(args.log_file))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("a", encoding="utf-8")
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _Tee(orig_stdout, log_f)
        sys.stderr = _Tee(orig_stderr, log_f)

        def _restore() -> None:
            try:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
            finally:
                try:
                    log_f.flush()
                except Exception:
                    pass
                log_f.close()

        atexit.register(_restore)

    mapping: dict[str, str] = {}
    if args.resume and mapping_path.exists():
        try:
            obj = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapping = obj if isinstance(obj, dict) else {}
        except Exception:
            mapping = {}

    if args.resume:
        removed = _cleanup_prediction_errors(preds_path=preds_path)
        if removed:
            print(f"resume cleanup removed_error_rows={removed} moved_to={preds_path.with_name('predictions.errors.jsonl')}")

    run_cat_sum = _new_cat_sums()
    run_cat_cnt = _new_cat_counts()
    run_judge_sum = _new_cat_sums()
    run_judge_cnt = _new_cat_counts()
    if args.judge and preds_path.exists() and (args.resume or args.summarize_only):
        _backfill_missing_judge(
            preds_path=preds_path,
            base_url=str(args.base_url),
            max_retries=int(args.max_retries),
            backoff_sec=float(args.retry_backoff_sec),
            concurrency=int(args.judge_concurrency),
        )
    if preds_path.exists():
        run_cat_sum, run_cat_cnt = _cat_f1_totals_from_predictions(preds_path)
        run_judge_sum, run_judge_cnt = _cat_judge_totals_from_predictions(preds_path)

    if args.summarize_only:
        summary = _compute_summary_from_predictions(
            preds_path=preds_path,
            run_id=run_id,
            base_url=str(args.base_url),
            samples=len(samples),
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    done_keys = _completed_keys_from_predictions(preds_path) if args.resume else set()

    if not args.resume:
        if preds_path.exists() and preds_path.stat().st_size > 0 and not args.overwrite:
            raise ValueError(f"Run already has results: {preds_path}. Use --resume or --overwrite.")
        preds_path.write_text("", encoding="utf-8")
        run_cat_sum = _new_cat_sums()
        run_cat_cnt = _new_cat_counts()
        run_judge_sum = _new_cat_sums()
        run_judge_cnt = _new_cat_counts()

    print(
        f"run start run_id={run_id} samples={len(samples)} base_url={args.base_url} "
        f"resume={bool(args.resume)} concurrency={int(args.concurrency)} outdir={outdir}"
    )
    if not args.skip_ingest and not bool(args.no_wait_embeddings):
        print("note: waiting for embeddings after ingest (use --no-wait-embeddings to disable).")

    with preds_path.open("a", encoding="utf-8") as preds_f, httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for sample_idx, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("sample_id") or sample.get("dia_id") or f"sample_{sample_idx}")
            user_id = mapping.get(sample_id) or _sanitize_user_id(f"{args.user_prefix}_{run_id}_{sample_id}")
            mapping[sample_id] = user_id
            mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            print(f"[{sample_idx}/{len(samples)}] sample start sample_id={sample_id} user_id={user_id}")
            turns = _extract_turns(sample)
            if not turns:
                continue

            if not args.skip_ingest:
                inserted_total = 0
                ignored_total = 0
                items = []
                for t in turns:
                    locomo = (t.meta or {}).get("locomo") if isinstance(t.meta, dict) else {}
                    session = (locomo or {}).get("session")
                    dia_id = (locomo or {}).get("dia_id")
                    mid = _stable_id_hex(sample_id, session, dia_id)
                    items.append(
                        {
                            "message_id": mid,
                            "ts": t.ts.isoformat(),
                            "role": t.role,
                            "content": t.content,
                            "meta": t.meta,
                        }
                    )
                for chunk in _chunk(items, 500):
                    resp = _post_json(client, f"/v1/users/{user_id}/messages:batch", payload={"items": chunk})
                    if isinstance(resp, dict):
                        inserted_total += int(resp.get("inserted") or 0)
                        ignored_total += int(resp.get("ignored") or 0)
                print(
                    f"ingest sample_id={sample_id} items={len(items)} inserted={inserted_total} ignored={ignored_total}"
                )
                if not bool(args.no_wait_embeddings):
                    try:
                        _wait_for_embeddings(
                            client,
                            user_id=user_id,
                            expected=int(inserted_total),
                            timeout_sec=float(args.embeddings_timeout_sec),
                        )
                    except TimeoutError as e:
                        print(f"warn: {e}")
            else:
                print(f"ingest sample_id={sample_id} skipped items={len(turns)}")

            if args.wait_embeddings_sec and args.wait_embeddings_sec > 0:
                time.sleep(float(args.wait_embeddings_sec))

            fallback_now = turns[-1].ts if turns else None
            qa_items = _extract_qa(sample, fallback_now=fallback_now)
            if only_cat is not None:
                qa_items = [qa for qa in qa_items if _parse_category_1_5(qa.category) == only_cat]
            if args.limit_qa is not None:
                qa_items = qa_items[: max(0, int(args.limit_qa))]

            already_done = sum(1 for qa in qa_items if (sample_id, int(qa.qa_idx)) in done_keys)
            remaining = len(qa_items) - already_done
            print(f"qa sample_id={sample_id} total={len(qa_items)} done={already_done} remaining={remaining} concurrency={int(args.concurrency)}")
            asyncio.run(
                _run_qa_concurrent(
                    base_url=args.base_url,
                    user_id=user_id,
                    sample_id=sample_id,
                    qa_items=qa_items,
                    preds_f=preds_f,
                    done_keys=done_keys,
                    concurrency=int(args.concurrency),
                    sleep_sec=float(args.sleep_sec or 0.0),
                    max_retries=int(args.max_retries),
                    backoff_sec=float(args.retry_backoff_sec),
                    progress_sec=float(args.progress_sec or 0.0),
                    progress_every=int(args.progress_every or 0),
                    cat_sum=run_cat_sum,
                    cat_cnt=run_cat_cnt,
                    judge_enabled=bool(args.judge),
                    judge_sum=run_judge_sum,
                    judge_cnt=run_judge_cnt,
                )
            )

    summary = _compute_summary_from_predictions(
        preds_path=preds_path,
        run_id=run_id,
        base_url=str(args.base_url),
        samples=len(samples),
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
