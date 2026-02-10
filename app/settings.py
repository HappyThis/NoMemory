from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"
    log_format: str = "json"  # json/text
    log_file: Optional[str] = None

    database_url: str = "postgresql+psycopg://nomemory:nomemory@127.0.0.1:5432/nomemory"
    cursor_secret: str = "dev-insecure"

    recall_skill: str = "nomemory-recall-default"
    # Recall agent hard budgets (context safety, not tool performance).
    recall_max_iterations: int = 3
    recall_max_tool_items: int = 50

    # LLM (agent reasoning) model selection
    llm_provider: str = "bigmodel"
    # NOTE: Prefer provider-specific model names below. Kept for backward compatibility.
    llm_model: str = ""

    # Provider-specific chat model names
    bigmodel_llm_model: str = "glm-4.7-flash"
    siliconflow_llm_model: str = "Pro/zai-org/GLM-4.7"

    # Embeddings provider (can be different from llm_provider)
    embedding_provider: str = "bigmodel"

    # BigModel (Zhipu) credentials/config. Endpoints are intentionally configurable so
    # the implementation can follow the latest vendor API without hard-coding URLs here.
    bigmodel_api_key: Optional[str] = None
    bigmodel_chat_endpoint: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    bigmodel_embedding_endpoint: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    bigmodel_embedding_model: str = "embedding-3"
    bigmodel_chat_timeout_sec: float = 180.0
    bigmodel_embedding_timeout_sec: float = 30.0

    # SiliconFlow (OpenAI-compatible chat completions)
    siliconflow_api_key: Optional[str] = None
    siliconflow_chat_endpoint: str = "https://api.siliconflow.cn/v1/chat/completions"
    siliconflow_chat_timeout_sec: float = 180.0


settings = Settings()
