"""Validate the LLM API and model."""

from typing import Any
from pydantic import Field, model_validator
from openai import OpenAI
from .api_types import ChatCompletion
from .api import LlmApiConfig


class LLManager(OpenAI):
    """Validate the LLM API and model."""

    api_config: dict[str, Any] = Field(default_factory=dict)
    llm_name: str = Field(default="gpt-4-0613")

    def __init__(
        self,
        llm_name: str = "gpt-4-0613",
        **kwargs: Any,
    ):
        self.llm_name = llm_name
        super().__init__(**kwargs)

    class Config:
        """
        Config for the Llamda class.
        """

        arbitrary_types_allowed = True

    def chat_completion(self, **kwargs: Any) -> ChatCompletion:
        """
        Override the chat.completions.create method.
        """
        return ChatCompletion.from_openai(super().chat.completions.create(**kwargs))

    @model_validator(mode="before")
    @classmethod
    def validate_api_and_model(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the API and model.
        """
        api_config = data.get("api_config") or {}
        api = (
            data.get("api")
            if isinstance(data.get("api"), OpenAI)
            else LlmApiConfig(**api_config).create_openai_client()
        )
        if not api or not isinstance(api, OpenAI):
            raise ValueError("Unable to create OpenAI client.")
        data.update({"api": api})

        if data.get("llm_name"):
            available_models: list[str] = [model.id for model in api.models.list()]
            if data.get("llm_name") not in available_models:
                raise ValueError(
                    f"Model '{data.get('llm_name')}' is not available. "
                    f"Available models: {', '.join(available_models)}"
                )
        else:
            raise ValueError("No LLM API client or LLM name provided.")

        return data


__all__: list[str] = ["LLManager"]
