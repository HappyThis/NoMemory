from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://nomemory:nomemory@127.0.0.1:5432/nomemory"
    cursor_secret: str = "dev-insecure"

    # Auth for ingest endpoint (service-to-service). In production use a real secret.
    ingest_api_key: str = "dev-ingest-key"

    # LLM (agent reasoning) model selection
    llm_provider: str = "bigmodel"
    llm_model: str = "glm-4.7-flash"

    # BigModel (Zhipu) credentials/config. Endpoints are intentionally configurable so
    # the implementation can follow the latest vendor API without hard-coding URLs here.
    bigmodel_api_key: Optional[str] = None
    bigmodel_chat_endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    bigmodel_embedding_endpoint: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    bigmodel_embedding_model: str = "embedding-3"


settings = Settings()
