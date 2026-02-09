from __future__ import annotations

import argparse
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
    args = ap.parse_args()

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
    results: list[dict[str, Any]] = []
    f1_by_cat: dict[str, list[float]] = defaultdict(list)
    scored = 0
    total = 0

    preds_path.write_text("", encoding="utf-8")
    with preds_path.open("a", encoding="utf-8") as preds_f, httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for sample_idx, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("sample_id") or sample.get("dia_id") or f"sample_{sample_idx}")
            user_id = _sanitize_user_id(f"{args.user_prefix}_{run_id}_{sample_id}")
            mapping[sample_id] = user_id

            print(f"[{sample_idx}/{len(samples)}] ingest sample_id={sample_id} user_id={user_id}")
            turns = _extract_turns(sample)
            if not turns:
                continue

            items = [{"ts": t.ts.isoformat(), "role": t.role, "content": t.content, "meta": t.meta} for t in turns]
            for chunk in _chunk(items, 500):
                _post_json(client, f"/v1/users/{user_id}/messages:batch", payload={"items": chunk})

            if args.wait_embeddings_sec and args.wait_embeddings_sec > 0:
                time.sleep(float(args.wait_embeddings_sec))

            fallback_now = turns[-1].ts if turns else None
            qa_items = _extract_qa(sample, fallback_now=fallback_now)
            if args.limit_qa is not None:
                qa_items = qa_items[: max(0, int(args.limit_qa))]

            for qa_idx, qa in enumerate(qa_items, start=1):
                total += 1
                headers = {"X-User-Id": user_id}
                payload: dict[str, Any] = {"question": qa.question}
                if qa.now is not None:
                    payload["now"] = qa.now.isoformat()

                pred_answer = ""
                evidence = []
                err = None
                try:
                    resp = _post_json(client, "/v1/bench/locomo/qa", headers=headers, payload=payload)
                    pred_answer = str(resp.get("answer") or "")
                    evidence = resp.get("evidence") or []
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"

                f1 = _max_f1(pred_answer, qa.answer)
                if f1 is not None:
                    scored += 1
                    cat = str(qa.category)
                    f1_by_cat[cat].append(float(f1))

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
                results.append(row)
                preds_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                preds_f.flush()

                if args.sleep_sec and args.sleep_sec > 0:
                    time.sleep(float(args.sleep_sec))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "run_id": run_id,
        "base_url": args.base_url,
        "samples": len(samples),
        "qa_total": total,
        "qa_scored": scored,
        "f1_mean": _mean([x for xs in f1_by_cat.values() for x in xs]),
        "f1_by_category": {k: {"count": len(v), "mean": _mean(v)} for (k, v) in sorted(f1_by_cat.items(), key=lambda kv: kv[0])},
        "predictions_path": str(preds_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
