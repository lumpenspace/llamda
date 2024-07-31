"""Module to handle the LLM APIs."""

from os import environ
from typing import Any, Optional
import dotenv

from pydantic import BaseModel, Field, field_validator
from .oai_api_types import OaiClient

dotenv.load_dotenv()

__all__: list[str] = [
    "LLApiConfig",
]


class LLApiConfig(BaseModel):
    """
    Configuration for the LLM API.
    """

    base_url: Optional[str] = None
    api_key: Optional[str] = Field(
        exclude=True,
        alias="api_key",
        default=environ.get("OPENAI_API_KEY"),
    )
    organization: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    default_headers: Optional[dict[str, Any]] = None
    default_query: Optional[dict[str, Any]] = None
    http_client: Optional[Any] = None  # You might want to use a more specific type here

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: Any) -> Optional[str]:
        """
        Validate the API key.
        """
        if not v and "base_url" not in info.data:
            raise ValueError("API key is required when base_url is not provided")
        return v

    def create_openai_client(self) -> OaiClient:
        """
        Create and return an OpenAI client with the configured settings.
        """
        config: dict[str, Any] = {
            k: v for k, v in self.model_dump().items() if v is not None
        }
        return OaiClient(**config)
