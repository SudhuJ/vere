from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
MODELS_FILE = PROJECT_ROOT / "models.json"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openai-compat"]
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class ModelRegistry(BaseModel):
    active: str
    models: Dict[str, ModelSpec]

    @property
    def active_model_spec(self) -> ModelSpec:
        return self.models[self.active]

    @model_validator(mode="after")
    def check_active_exists(self):
        if self.active not in self.models:
            raise ValueError(f"active model '{self.active}' not in registry")
        return self


def load_registry() -> ModelRegistry:
    return ModelRegistry.model_validate_json(MODELS_FILE.read_text())
