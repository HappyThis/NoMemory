from fastapi import FastAPI

from app.api.bench_locomo import router as locomo_bench_router
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.api.recall import router as recall_router
from app.logging_setup import RequestIdMiddleware, setup_logging


setup_logging()
app = FastAPI(title="NoMemory")
app.add_middleware(RequestIdMiddleware)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(recall_router)
app.include_router(locomo_bench_router)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
