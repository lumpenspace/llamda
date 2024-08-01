"""Manages communications to and from APIs."""

from typing import Any
from pydantic import BaseModel, Field, model_validator, ConfigDict

from llamda_fn.llms.ll_exchange import LLExchange
from llamda_fn.llms.oai_api_types import OaiClient, OaiException
from llamda_fn.llms.ll_message import LLMessage
from .ll_api_config import LLApiConfig

__all__ = ["LLManager"]


class LLManager(BaseModel):
    """A client and manager for OAI-like LLM APIs"""

    api_config: dict[str, Any] = Field(default={})
    llm_name: str = Field(default="gpt-4o-mini")
    api: OaiClient | None = None
    _available_llms = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def query(
        self, messages: LLExchange, llm_name: str | None, **kwargs: Any
    ) -> LLMessage:
        """
        Sends the messages to the OpenAI API and returns the response.

        Args:
            messages (LLExchange): The exchange of messages to send to the API.
            llm_name (str | None): The name of the language model to use.
            If None, uses the default model.
            **kwargs: Additional keyword arguments to pass to the API call.

        Returns:
            LLMessage: The response message from the API.

        Raises:
            OaiException: If there's an error in the chat completion.
            NameError: If the specified LLM is not available.
        """
        try:
            if llm_name and llm_name not in self._available_llms:
                raise NameError(
                    f"Unavailable LLM: {llm_name}",
                    "Available models are: {self._available_llms}",
                )
            if self.api is None:
                raise ValueError("API client is not initialized.")

            return LLMessage.from_completion(
                self.api.chat.completions.create(
                    messages=messages.oai_props,
                    model=llm_name or self.llm_name,
                    **kwargs,
                )
            )
        except Exception as e:
            raise OaiException(f"Error in chat completion: {str(e)}", messages) from e

    @model_validator(mode="before")
    @classmethod
    def validate_api_and_llm(cls, data: Any) -> Any:
        """Validate the API and model."""

        api_config: Any | dict[Any, Any] = data.get("api_config") or {}
        api: Any | OaiClient = (
            data.get("api")
            if isinstance(data.get("api"), OaiClient)
            else LLApiConfig(**api_config).create_openai_client()
        )
        if not api or not isinstance(api, OaiClient):
            raise ValueError("Unable to create OpenAI client.")

        data.update({"api": api})

        available_models: set[str] = {model.id for model in api.models.list()}
        cls._available_llms: list[str] = list(available_models)
        data.update({"_available_llms": cls._available_llms})
        if data.get("llm_name") not in available_models:
            raise ValueError(
                f"Model '{data.get('llm_name')}' is not available. "
                f"Available models: {', '.join(cls._available_llms)}"
            )
        return data
