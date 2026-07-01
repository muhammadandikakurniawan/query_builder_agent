from  pathlib import Path
from typing import Optional

import yaml
from  pydantic import BaseModel, Field

from agent_app.shared.database.connection.model import DatabaseConnectionConfig


class LLMProviderConfig(BaseModel):
    name: str

    base_url: str
    api_key: str

    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = 60

class HTTPConfig(BaseModel):
    host: str
    port: int
    api_timeout: Optional[int] = None


class EmbeddingConfig(BaseModel):
    embedding_model: Optional[str] = 'BAAI/bge-large-en-v1.5'
    default_truncate_dim: Optional[int] = 1024

class LLMConfig(BaseModel):
    providers: list[LLMProviderConfig] = Field(default_factory=list)

    def get_provider(self,  name: str) -> LLMProviderConfig:
        for model in self.providers:
            if model.name == name:
                return model

        raise ValueError(
            f"LLM model '{name}' not found"
        )

class GoogleDrive(BaseModel):
    acess_token: str
    dir_id: str
    creds: str

class App(BaseModel):
    debug: bool
    log_level: str
    http_server: HTTPConfig

class Callback(BaseModel):
    url: str
    timeout_in_second: int

class DatabaseConfig(BaseModel):
    master: DatabaseConnectionConfig
    slaves: list[DatabaseConnectionConfig] = []

class VectorDbConfig(BaseModel):
    host: str
    port: int

class HuggingFaceConfig(BaseModel):
    token: Optional[str] = None

class Config(BaseModel):
    app: App
    llm: Optional[LLMConfig] = None
    google_drive: Optional[GoogleDrive] = None
    upload_file_callback: Optional[Callback] = None
    database: DatabaseConfig
    qdrant: VectorDbConfig
    embedding: Optional[EmbeddingConfig] = EmbeddingConfig()
    huggingface: Optional[HuggingFaceConfig] = HuggingFaceConfig()



# Singleton instance
_config: Config | None = None


def load_config(path: str = "config.yaml") -> Config:
    """
    Load config.yaml and initialize singleton.
    Should be called once during startup.
    """
    global _config

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _config = Config.model_validate(data)
    return _config


def get_config() -> Config:
    """
    Retrieve singleton config instance.
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() first."
        )

    return _config