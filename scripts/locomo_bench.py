from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import random
import string
import time
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
    s = int(max(0.0, float(seconds)))
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
    question: str
    answer: Any
    category: Any
    evidence: Any
    now: Optional[datetime]


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
    for item in qa:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        if not isinstance(q, str) or not q.strip():
            continue
        now = _parse_dt(item.get("now")) or _parse_dt(item.get("query_time")) or _parse_dt(item.get("query_date_time")) or fallback_now
        out.append(
            QAItem(
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


def _compute_summary_from_predictions(*, preds_path: Path, run_id: str, base_url: str, samples: int) -> dict[str, Any]:
    f1_by_cat: dict[str, list[float]] = defaultdict(list)
    total = 0
    scored = 0
    for obj in _safe_load_jsonl(preds_path):
        total += 1
        f1 = obj.get("f1")
        if isinstance(f1, (int, float)):
            scored += 1
            cat = str(obj.get("category"))
            f1_by_cat[cat].append(float(f1))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "run_id": run_id,
        "base_url": base_url,
        "samples": samples,
        "qa_total": total,
        "qa_scored": scored,
        "f1_mean": _mean([x for xs in f1_by_cat.values() for x in xs]),
        "f1_by_category": {k: {"count": len(v), "mean": _mean(v)} for (k, v) in sorted(f1_by_cat.items(), key=lambda kv: kv[0])},
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
) -> None:
    write_lock = asyncio.Lock()
    tasks: list[tuple[int, QAItem]] = []
    for qa_idx, qa in enumerate(qa_items, start=1):
        if (sample_id, qa_idx) not in done_keys:
            tasks.append((qa_idx, qa))

    if not tasks:
        print(f"qa sample_id={sample_id} all_done total={len(qa_items)}")
        return

    total_to_run = len(tasks)
    completed = 0
    errors = 0
    last_idx: int | None = None
    t0 = time.monotonic()

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
                    f"rate={rate:.2f}/s eta={_fmt_eta(eta)} last={li}"
                )

        async def worker() -> None:
            nonlocal completed, errors, last_idx
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

                pred_answer = ""
                evidence = []
                err = None
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

                f1 = _max_f1(pred_answer, qa.answer)
                row = {
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "qa_idx": qa_idx,
                    "category": qa.category,
                    "question": qa.question,
                    "gold_answer": qa.answer,
                    "pred_answer": pred_answer,
                    "f1": f1,
                    "now": qa.now.isoformat() if qa.now else None,
                    "evidence_count": len(evidence) if isinstance(evidence, list) else None,
                    "error": err,
                }

                async with write_lock:
                    preds_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    preds_f.flush()
                    done_keys.add((sample_id, qa_idx))
                    completed += 1
                    if progress_every and completed % int(progress_every) == 0:
                        elapsed = max(1e-6, time.monotonic() - t0)
                        rate = completed / elapsed
                        remaining = total_to_run - completed
                        eta = (remaining / rate) if rate > 0 else float("inf")
                        print(
                            f"qa sample_id={sample_id} done={completed}/{total_to_run} errors={errors} "
                            f"rate={rate:.2f}/s eta={_fmt_eta(eta)} last={qa_idx}"
                        )

                if sleep_sec and sleep_sec > 0:
                    await asyncio.sleep(float(sleep_sec))
                q.task_done()

        report_task = asyncio.create_task(reporter())
        workers = [asyncio.create_task(worker()) for _ in range(max(1, int(concurrency)))]
        await q.join()
        stop.set()
        await report_task
        for w in workers:
            await w

    elapsed = max(1e-6, time.monotonic() - t0)
    print(f"qa sample_id={sample_id} done={completed}/{total_to_run} errors={errors} elapsed={_fmt_eta(elapsed)}")


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
    ap.add_argument("--wait-embeddings-sec", type=float, default=0.0, help="Wait after ingest to let embeddings finish.")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run (requires --run-id; appends to predictions.jsonl).")
    ap.add_argument("--concurrency", type=int, default=2, help="Number of concurrent QA requests (default: 2).")
    ap.add_argument("--max-retries", type=int, default=3, help="Max retries per QA request on retryable errors (default: 3).")
    ap.add_argument("--retry-backoff-sec", type=float, default=1.0, help="Base backoff seconds for retries (default: 1.0).")
    ap.add_argument("--skip-ingest", action="store_true", help="Skip ingest and only run QA (useful with --resume).")
    ap.add_argument("--progress-sec", type=float, default=5.0, help="Print progress every N seconds during QA (default: 5).")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N completed QA (default: 10).")
    args = ap.parse_args()

    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id")
    if args.concurrency is None or int(args.concurrency) <= 0:
        raise ValueError("--concurrency must be >= 1")
    if args.max_retries is None or int(args.max_retries) < 0:
        raise ValueError("--max-retries must be >= 0")
    if args.retry_backoff_sec is None or float(args.retry_backoff_sec) <= 0:
        raise ValueError("--retry-backoff-sec must be > 0")

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

    mapping: dict[str, str] = {}
    if args.resume and mapping_path.exists():
        try:
            obj = json.loads(mapping_path.read_text(encoding="utf-8"))
            mapping = obj if isinstance(obj, dict) else {}
        except Exception:
            mapping = {}

    done_keys = _completed_keys_from_predictions(preds_path) if args.resume else set()

    if not args.resume:
        preds_path.write_text("", encoding="utf-8")

    print(
        f"run start run_id={run_id} samples={len(samples)} base_url={args.base_url} "
        f"resume={bool(args.resume)} concurrency={int(args.concurrency)} outdir={outdir}"
    )

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
            else:
                print(f"ingest sample_id={sample_id} skipped items={len(turns)}")

            if args.wait_embeddings_sec and args.wait_embeddings_sec > 0:
                time.sleep(float(args.wait_embeddings_sec))

            fallback_now = turns[-1].ts if turns else None
            qa_items = _extract_qa(sample, fallback_now=fallback_now)
            if args.limit_qa is not None:
                qa_items = qa_items[: max(0, int(args.limit_qa))]

            already_done = sum(1 for i in range(1, len(qa_items) + 1) if (sample_id, i) in done_keys)
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
