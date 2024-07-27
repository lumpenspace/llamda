from typing import Any
from pydantic import BaseModel, Field, model_validator
from openai import OpenAI
from openai.types.model import Model as OpenAIModel
from llamda_fn.utils import LlmApiConfig


class LlamdaValidator(BaseModel):
    """Validate the LLM API and model."""

    api: OpenAI | None = Field(default=None)
    api_config: dict[str, Any] = Field(default_factory=dict)
    llm_name: str = Field(default="gpt-4-0613")

    class Config:
        """
        Config for the Llamda class.
        """

        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_api_and_model(cls, data: dict[str, Any]) -> dict[str, Any]:
        api_config = data.get("api_config") or {}
        api = (
            data.get("api")
            if isinstance(data.get("api"), OpenAI)
            else LlmApiConfig(**api_config).create_openai_client()
        )
        if not api or not isinstance(api, OpenAI):
            raise ValueError("Unable to create OpenAI client.")
        data.update({"api": api})
        return data

    @model_validator(mode="after")
    def instance_validator(self) -> "LlamdaValidator":
        """
        Mostly makes sure that the API is there.
        """
        if not self.api:
            raise ValueError("No LLM API client provided.")
        if self.llm_name and self.api:
            available_models: list[OpenAIModel] = list(self.api.models.list())
            model_ids = [model.id for model in available_models]
            if self.llm_name not in model_ids:
                raise ValueError(
                    f"Model '{self.llm_name}' is not available. "
                    f"Available models: {', '.join(model_ids)}"
                )
        else:
            raise ValueError("No LLM API client or LLM name provided.")

        return self


__all__: list[str] = ["LlamdaValidator"]
