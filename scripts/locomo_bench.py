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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

DEFAULT_LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
FIXED_USER_PREFIX = "locomo"


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


def _wait_for_embeddings(client: httpx.Client, *, user_id: str) -> None:
    t0 = time.monotonic()
    sleep_s = 0.25
    last_emb = -1
    last_msgs = -1
    last_print = 0.0
    enqueued = False
    while True:
        status = _get_json(client, f"/v1/users/{user_id}/embeddings/status")
        enabled = bool(status.get("enabled"))
        msgs = int(status.get("messages") or 0)
        emb = int(status.get("embeddings") or 0)
        if not enabled:
            return
        if msgs <= 0:
            return
        if emb >= msgs:
            return

        elapsed = time.monotonic() - t0

        # If this user already has messages but embeddings are behind (common in resume where inserted=0),
        # try to (re)enqueue missing embeddings once so we don't wait forever on a stalled background task.
        if not enqueued:
            try:
                _post_json(client, f"/v1/users/{user_id}/embeddings/enqueue", payload={})
                enqueued = True
            except Exception:
                # Best-effort; keep waiting.
                enqueued = True

        if emb != last_emb or msgs != last_msgs:
            last_emb = emb
            last_msgs = msgs
            last_print = time.monotonic()
            print(f"wait embeddings user_id={user_id} embeddings={emb}/{msgs} elapsed={_fmt_eta(elapsed)}")
        elif (time.monotonic() - last_print) >= 30.0:
            last_print = time.monotonic()
            print(f"wait embeddings user_id={user_id} embeddings={emb}/{msgs} elapsed={_fmt_eta(elapsed)}")
        time.sleep(sleep_s)
        sleep_s = min(2.0, sleep_s * 1.5)


async def _post_json_async(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any],
) -> Any:
    r = await client.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()


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


def _parse_metrics_arg(v: Any) -> set[str]:
    """
    Parse --metrics.

    Supported:
      - f1: token-level F1
      - llm: LLM-as-judge (requires /v1/bench/locomo/judge)
    """
    if v is None:
        return {"f1"}
    parts = re.split(r"[,\s]+", str(v))
    metrics: set[str] = set()
    for p in parts:
        s = p.strip().lower()
        if not s:
            continue
        if s in ("judge", "llm_judge", "llm-as-judge", "llm_as_judge"):
            s = "llm"
        metrics.add(s)
    if not metrics:
        metrics = {"f1"}
    allowed = {"f1", "llm"}
    unknown = sorted(m for m in metrics if m not in allowed)
    if unknown:
        raise ValueError(f"--metrics invalid: {unknown} (allowed: f1,llm)")
    return metrics


def _row_has_required_metrics(obj: dict[str, Any], *, required: set[str]) -> bool:
    err = obj.get("error")
    if err is not None and (not isinstance(err, str) or err.strip()):
        return False

    if "f1" in required and "f1" not in obj:
        return False

    if "llm" in required:
        label = obj.get("judge_label")
        score = obj.get("judge_score")
        if isinstance(score, bool):
            return True
        if isinstance(score, int) and score in (0, 1):
            return (label in (None, "CORRECT", "WRONG")) if label is not None else True
        if isinstance(score, float) and score in (0.0, 1.0):
            return (label in (None, "CORRECT", "WRONG")) if label is not None else True
        return False

    return True


def _completed_keys_from_predictions(path: Path, *, required_metrics: set[str]) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    for obj in _safe_load_jsonl(path):
        sample_id = obj.get("sample_id")
        qa_idx = obj.get("qa_idx")
        if isinstance(sample_id, str) and isinstance(qa_idx, int):
            if _row_has_required_metrics(obj, required=required_metrics):
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
        # LoCoMo category ids in locomo10.json follow the repository evaluation code, not the paper's numbered list:
        #   1=Multi-hop, 2=Temporal, 3=Open-domain, 4=Single-hop, 5=Adversarial
        "1": 1,
        "multi": 1,
        "multi-hop": 1,
        "multihop": 1,
        "2": 2,
        "temporal": 2,
        "time": 2,
        "3": 3,
        "open": 3,
        "open-domain": 3,
        "open_domain": 3,
        "opendomain": 3,
        # Some write-ups refer to this bucket as "open-domain/commonsense".
        "commonsense": 3,
        "common-sense": 3,
        "4": 4,
        "single": 4,
        "single-hop": 4,
        "singlehop": 4,
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
    1: "Multi-hop",
    2: "Temporal",
    3: "Open-domain",
    4: "Single-hop",
    5: "Adversarial",
}

_EVALUATED_CATEGORY_IDS: tuple[int, ...] = (1, 2, 3, 4)

def _score_f1(*, pred_answer: str, gold_answer: Any, category: Any) -> Optional[float]:
    """
    Token-level F1 against LoCoMo gold answers.

    Note: Adversarial questions (category=5) are not evaluated by this runner.
    """
    return _max_f1(pred_answer, gold_answer)


def _new_cat_sums() -> dict[int, float]:
    return {i: 0.0 for i in _EVALUATED_CATEGORY_IDS}


def _new_cat_counts() -> dict[int, int]:
    return {i: 0 for i in _EVALUATED_CATEGORY_IDS}


def _fmt_cat_f1_means(cat_sum: dict[int, float], cat_cnt: dict[int, int]) -> str:
    parts: list[str] = []
    for i in _EVALUATED_CATEGORY_IDS:
        cnt = int(cat_cnt.get(i) or 0)
        mean = (float(cat_sum.get(i) or 0.0) / cnt) if cnt > 0 else 0.0
        parts.append(f"{_CATEGORY_LABELS.get(i, str(i))}={mean:.3f}")
    return " ".join(parts)


def _fmt_cat_judge_means(judge_sum: dict[int, float], judge_cnt: dict[int, int]) -> str:
    parts: list[str] = []
    for i in _EVALUATED_CATEGORY_IDS:
        cnt = int(judge_cnt.get(i) or 0)
        mean = (float(judge_sum.get(i) or 0.0) / cnt) if cnt > 0 else 0.0
        parts.append(f"{_CATEGORY_LABELS.get(i, str(i))}={mean:.3f}")
    return " ".join(parts)


def _cat_f1_totals_from_predictions(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    """
    Build run-level category (1..4) F1 sum/count from predictions.jsonl.

    If a row has f1==null/missing, it is treated as 0 for aggregation (no recomputation).
    If there are duplicate (sample_id, qa_idx) rows, the last one wins.
    """
    by_key: dict[tuple[str, int], tuple[int, float]] = {}
    for obj in _safe_load_jsonl(path):
        sample_id = obj.get("sample_id")
        qa_idx = obj.get("qa_idx")
        if not (isinstance(sample_id, str) and isinstance(qa_idx, int)):
            continue
        cat = _parse_category_1_5(obj.get("category"))
        if cat not in _EVALUATED_CATEGORY_IDS:
            continue
        f1v = obj.get("f1")
        f1 = float(f1v) if isinstance(f1v, (int, float)) and math.isfinite(float(f1v)) else 0.0
        by_key[(sample_id, qa_idx)] = (cat, f1)

    cat_sum = _new_cat_sums()
    cat_cnt = _new_cat_counts()
    for _, (cat, f1) in by_key.items():
        cat_sum[cat] += float(f1)
        cat_cnt[cat] += 1
    return cat_sum, cat_cnt


def _cat_judge_totals_from_predictions(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    """
    Build run-level category (1..4) judge score sum/count from predictions.jsonl.

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
        if cat not in _EVALUATED_CATEGORY_IDS:
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
    # Deduplicate by (sample_id, qa_idx); last row wins (important for resume).
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for obj in _safe_load_jsonl(preds_path):
        sample_id = obj.get("sample_id")
        qa_idx = obj.get("qa_idx")
        if isinstance(sample_id, str) and isinstance(qa_idx, int):
            by_key[(sample_id, qa_idx)] = obj

    # If there are migrated error rows (resume moved them out for retry) but they haven't been
    # successfully re-run yet, include them in the denominator as incorrect.
    errors_path = preds_path.with_name("predictions.errors.jsonl")
    if errors_path.exists():
        for obj in _safe_load_jsonl(errors_path):
            sample_id = obj.get("sample_id")
            qa_idx = obj.get("qa_idx")
            if isinstance(sample_id, str) and isinstance(qa_idx, int):
                by_key.setdefault((sample_id, qa_idx), obj)

    f1_by_cat: dict[str, list[float]] = defaultdict(list)
    judge_by_cat: dict[str, list[float]] = defaultdict(list)
    total = 0
    error_rows = 0
    f1_scored = 0
    judge_scored = 0

    f1_enabled = any("f1" in obj for obj in by_key.values())
    judge_enabled = any(
        ("judge_score" in obj) or ("judge_label" in obj) or ("judge_reason" in obj) or ("judge_request_id" in obj)
        for obj in by_key.values()
    )

    for obj in by_key.values():
        cat_i = _parse_category_1_5(obj.get("category"))
        if cat_i not in _EVALUATED_CATEGORY_IDS:
            continue
        total += 1
        cat = _CATEGORY_LABELS.get(cat_i, str(cat_i))

        err = obj.get("error")
        if err is not None:
            if not isinstance(err, str):
                error_rows += 1
            elif err.strip():
                error_rows += 1

        if f1_enabled:
            f1v = obj.get("f1") if "f1" in obj else None
            if isinstance(f1v, (int, float)) and math.isfinite(float(f1v)):
                f1_scored += 1
                f1_by_cat[cat].append(float(f1v))
            else:
                # Treat missing / invalid / errored rows as incorrect (0.0) in the denominator.
                f1_by_cat[cat].append(0.0)

        if judge_enabled:
            js = obj.get("judge_score")
            if isinstance(js, bool):
                js = 1.0 if js else 0.0
            if isinstance(js, (int, float)) and math.isfinite(float(js)):
                judge_scored += 1
                judge_by_cat[cat].append(float(js))
            else:
                # Treat missing / invalid / errored rows as incorrect (0.0) in the denominator.
                judge_by_cat[cat].append(0.0)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    label_order = {v: k for (k, v) in _CATEGORY_LABELS.items()}
    def _sort_key(k: str) -> tuple[int, str]:
        return (label_order.get(k, 999), k)

    f1_total = total if f1_enabled else 0
    judge_total = total if judge_enabled else 0

    return {
        "run_id": run_id,
        "base_url": base_url,
        "samples": samples,
        "qa_total": total,
        "qa_errors": error_rows,
        "qa_error_rate": (error_rows / total if total else 0.0),
        "qa_scored": f1_scored,
        "f1_total": f1_total,
        "f1_scored": f1_scored,
        "f1_mean": (_mean([x for xs in f1_by_cat.values() for x in xs]) if f1_by_cat else None),
        "f1_by_category": ({k: {"count": len(v), "mean": _mean(v)} for (k, v) in sorted(f1_by_cat.items(), key=lambda kv: _sort_key(kv[0]))} if f1_by_cat else None),
        "f1_missing": (f1_total - f1_scored) if f1_enabled else None,
        "judge_total": judge_total,
        "judge_scored": judge_scored,
        "judge_mean": (_mean([x for xs in judge_by_cat.values() for x in xs]) if judge_by_cat else None),
        "judge_by_category": ({k: {"count": len(v), "mean": _mean(v)} for (k, v) in sorted(judge_by_cat.items(), key=lambda kv: _sort_key(kv[0]))} if judge_by_cat else None),
        "judge_missing": (judge_total - judge_scored) if judge_enabled else None,
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
    f1_enabled: bool,
    cat_sum: dict[int, float],
    cat_cnt: dict[int, int],
    llm_enabled: bool,
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

    q: asyncio.Queue[tuple[int, QAItem] | None] = asyncio.Queue()
    for t in tasks:
        q.put_nowait(t)

    for _ in range(max(1, int(concurrency))):
        q.put_nowait(None)

    headers = {"X-User-Id": user_id}
    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        def _fmt_err(e: Exception) -> str:
            if isinstance(e, httpx.HTTPStatusError):
                rid = e.response.headers.get("X-Request-Id") or "-"
                return f"HTTP {e.response.status_code} request_id={rid}"
            return f"{type(e).__name__}: {e}"

        async def worker() -> None:
            nonlocal completed, errors, last_idx, in_flight
            while True:
                item = await q.get()
                if item is None:
                    q.task_done()
                    return
                qa_idx, qa = item
                last_idx = qa_idx

                total_payload: dict[str, Any] = {"question": qa.question}
                if qa.now is not None:
                    total_payload["now"] = qa.now.isoformat()

                qa_request_id = f"{user_id}__qa__{qa_idx}"
                qa_headers = dict(headers)
                qa_headers["X-Request-Id"] = qa_request_id

                pred_answer = ""
                evidence = []
                f1: float | None = None
                judge_label: str | None = None
                judge_score: int | None = None
                judge_reason: str | None = None
                judge_request_id: str | None = None
                err = None
                in_flight += 1
                try:
                    try:
                        resp = await _post_json_async(
                            client,
                            "/v1/bench/locomo/qa",
                            headers=qa_headers,
                            payload=total_payload,
                        )
                        pred_answer = str(resp.get("answer") or "")
                        evidence = resp.get("evidence") or []
                    except Exception as e:
                        err = _fmt_err(e)
                        errors += 1

                    if err is None and f1_enabled:
                        f1 = _score_f1(pred_answer=pred_answer, gold_answer=qa.answer, category=qa.category)
                    if err is None and llm_enabled:
                        try:
                            judge_request_id = f"{user_id}__judge__{qa_idx}"
                            judge_headers = dict(headers)
                            judge_headers["X-Request-Id"] = judge_request_id
                            jresp = await _post_json_async(
                                client,
                                "/v1/bench/locomo/judge",
                                headers=judge_headers,
                                payload={
                                    "question": qa.question,
                                    "gold_answer": qa.answer,
                                    "pred_answer": pred_answer,
                                    "category": _parse_category_1_5(qa.category),
                                    "now": qa.now.isoformat() if qa.now else None,
                                },
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
                            err = "JudgeError " + _fmt_err(e)
                            errors += 1
                finally:
                    in_flight -= 1
                row = {
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "qa_idx": qa_idx,
                    "request_id": qa_request_id,
                    "category": qa.category,
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "pred_answer": pred_answer,
                    "now": qa.now.isoformat() if qa.now else None,
                    "evidence_count": len(evidence) if isinstance(evidence, list) else None,
                    "error": err,
                }
                if f1_enabled:
                    row["f1"] = f1
                if llm_enabled:
                    row["judge_request_id"] = judge_request_id
                    row["judge_label"] = judge_label
                    row["judge_score"] = judge_score
                    row["judge_reason"] = judge_reason

                async with write_lock:
                    preds_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    preds_f.flush()
                    done_keys.add((sample_id, qa_idx))
                    cat = _parse_category_1_5(qa.category)
                    if cat in _EVALUATED_CATEGORY_IDS:
                        if f1_enabled:
                            cat_sum[cat] = float(cat_sum.get(cat) or 0.0) + (float(f1) if isinstance(f1, (int, float)) else 0.0)
                            cat_cnt[cat] = int(cat_cnt.get(cat) or 0) + 1
                        if llm_enabled and judge_score is not None:
                            judge_sum[cat] = float(judge_sum.get(cat) or 0.0) + float(judge_score)
                            judge_cnt[cat] = int(judge_cnt.get(cat) or 0) + 1
                    completed += 1

                q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(max(1, int(concurrency)))]
        await q.join()
        for w in workers:
            await w

    elapsed = max(1e-6, time.monotonic() - t0)
    print(
        f"qa sample_id={sample_id} done={completed}/{total_to_run} errors={errors} elapsed={_fmt_eta(elapsed)} "
        + (f"f1_mean_by_cat={_fmt_cat_f1_means(cat_sum, cat_cnt)} " if f1_enabled else "")
        + (f"judge_mean_by_cat={_fmt_cat_judge_means(judge_sum, judge_cnt)}" if llm_enabled else "")
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
    ap = argparse.ArgumentParser(description="LoCoMo benchmark runner: ingest + QA + metrics report")
    ap.add_argument(
        "--data",
        default=None,
        help="Path to an existing locomo10.json. If omitted or not found, it will be downloaded to ./data/locomo/locomo10.json.",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8001", help="NoMemory service base URL.")
    ap.add_argument("--outdir", default="traces/locomo", help="Output directory for results.")
    ap.add_argument("--run-id", default=None, help="Resume an existing run id under --outdir (must already exist).")
    ap.add_argument("--concurrency", type=int, default=2, help="Number of concurrent QA requests (default: 2).")
    ap.add_argument("--metrics", default="f1,llm", help="Comma/space-separated metrics to compute during evaluation: f1,llm (default: f1,llm).")
    ap.add_argument("--only-category", default=None, help="Run only one QA category (1-4 or name, e.g. temporal). Adversarial is disabled.")
    ap.add_argument("--log-file", default=None, help="Append stdout/stderr logs to this file.")
    ap.add_argument("--summarize-only", action="store_true", help="Only (re)write summary.json from predictions.jsonl; do not ingest or run QA.")
    args = ap.parse_args()

    resume_mode = bool(args.run_id)
    if args.summarize_only and not resume_mode:
        raise ValueError("--summarize-only requires --run-id")
    if args.concurrency is None or int(args.concurrency) <= 0:
        raise ValueError("--concurrency must be >= 1")
    metrics = _parse_metrics_arg(args.metrics)
    only_cat = _parse_only_category_arg(args.only_category)
    if args.only_category is not None and only_cat is None:
        raise ValueError(f"--only-category invalid: {args.only_category!r} (expected 1-4 or a name like temporal)")
    if only_cat == 5:
        raise ValueError("--only-category adversarial is not supported (Adversarial is disabled in this runner).")

    default_data_path = Path("data/locomo/locomo10.json")
    data_path = Path(args.data) if (args.data and Path(args.data).exists()) else default_data_path
    if not data_path.exists():
        print(f"download locomo10.json url={DEFAULT_LOCOMO_URL} dest={default_data_path}")
        _download_file(url=DEFAULT_LOCOMO_URL, dest=default_data_path)
        data_path = default_data_path

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    samples = _extract_locomo_samples(raw)

    run_id = str(args.run_id) if resume_mode else _now_utc().strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) / run_id
    if resume_mode:
        if not outdir.exists():
            raise FileNotFoundError(f"--run-id not found under --outdir: {outdir}")
        if not outdir.is_dir():
            raise NotADirectoryError(f"--run-id is not a directory: {outdir}")
    else:
        if outdir.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {outdir}")
        outdir.mkdir(parents=True, exist_ok=False)

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
    if resume_mode:
        if not preds_path.exists():
            raise FileNotFoundError(f"Missing predictions file for --run-id: {preds_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing user mapping file for --run-id: {mapping_path}")
        obj = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid user mapping file (expected JSON object): {mapping_path}")
        mapping = {str(k): str(v) for (k, v) in obj.items()}

        if not args.summarize_only:
            removed = _cleanup_prediction_errors(preds_path=preds_path)
            if removed:
                print(
                    f"resume cleanup removed_error_rows={removed} moved_to={preds_path.with_name('predictions.errors.jsonl')}"
                )

    run_cat_sum = _new_cat_sums()
    run_cat_cnt = _new_cat_counts()
    run_judge_sum = _new_cat_sums()
    run_judge_cnt = _new_cat_counts()
    if preds_path.exists():
        if "f1" in metrics:
            run_cat_sum, run_cat_cnt = _cat_f1_totals_from_predictions(preds_path)
        if "llm" in metrics:
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

    done_keys = _completed_keys_from_predictions(preds_path, required_metrics=metrics) if resume_mode else set()

    if not resume_mode:
        preds_path.write_text("", encoding="utf-8")
        mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run_cat_sum = _new_cat_sums()
        run_cat_cnt = _new_cat_counts()
        run_judge_sum = _new_cat_sums()
        run_judge_cnt = _new_cat_counts()

    print(
        f"run start run_id={run_id} samples={len(samples)} base_url={args.base_url} "
        f"concurrency={int(args.concurrency)} outdir={outdir}"
    )
    print(f"metrics={','.join(sorted(metrics))} (computed during evaluation)")

    with preds_path.open("a", encoding="utf-8") as preds_f, httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for sample_idx, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("sample_id") or sample.get("dia_id") or f"sample_{sample_idx}")
            user_id = mapping.get(sample_id) or _sanitize_user_id(f"{FIXED_USER_PREFIX}_{run_id}_{sample_id}")
            mapping[sample_id] = user_id
            mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            print(f"[{sample_idx}/{len(samples)}] sample start sample_id={sample_id} user_id={user_id}")
            turns = _extract_turns(sample)
            if not turns:
                continue

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
            print(f"ingest sample_id={sample_id} items={len(items)} inserted={inserted_total} ignored={ignored_total}")

            _wait_for_embeddings(client, user_id=user_id)

            fallback_now = turns[-1].ts if turns else None
            qa_items = _extract_qa(sample, fallback_now=fallback_now)
            qa_items = [qa for qa in qa_items if _parse_category_1_5(qa.category) in _EVALUATED_CATEGORY_IDS]
            if only_cat is not None:
                qa_items = [qa for qa in qa_items if _parse_category_1_5(qa.category) == only_cat]

            already_done = sum(1 for qa in qa_items if (sample_id, int(qa.qa_idx)) in done_keys)
            remaining = len(qa_items) - already_done
            print(
                f"qa sample_id={sample_id} total={len(qa_items)} done={already_done} remaining={remaining} concurrency={int(args.concurrency)}"
            )
            asyncio.run(
                _run_qa_concurrent(
                    base_url=args.base_url,
                    user_id=user_id,
                    sample_id=sample_id,
                    qa_items=qa_items,
                    preds_f=preds_f,
                    done_keys=done_keys,
                    concurrency=int(args.concurrency),
                    f1_enabled=("f1" in metrics),
                    cat_sum=run_cat_sum,
                    cat_cnt=run_cat_cnt,
                    llm_enabled=("llm" in metrics),
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
