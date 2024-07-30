from typing import Any, Self
from pydantic import Field, model_validator
from openai import OpenAI
from openai.types.chat import ChatCompletion

from llamda_fn.llms.exchange import Exchange
from .api_types import LLCompletion
from .api import LlmApiConfig
from llamda_fn.utils.logger import logger


class LLManager(OpenAI):
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
        arbitrary_types_allowed = True

    def chat_completion(
        self, messages: Exchange, llm_name: str, **kwargs: Any
    ) -> LLCompletion:
        oai_messages = []
        for message in messages:
            oai_message = message.get_oai_message()
            oai_messages.append(
                {
                    "role": oai_message["role"],
                    "content": oai_message["content"],
                }
            )
            if oai_message.get("name"):
                oai_messages[-1]["name"] = oai_message["name"]
            if oai_message.get("tool_calls"):
                oai_messages[-1]["tool_calls"] = oai_message["tool_calls"]

        try:
            print(messages)
            oai_completion: ChatCompletion = self.chat.completions.create(
                messages=oai_messages,
                model=llm_name or self.llm_name,
                **kwargs,
            )
            return LLCompletion.from_completion(oai_completion)
        except Exception as e:
            raise Exception(f"Error in chat completion: {str(e)}", messages) from e

    @model_validator(mode="after")
    def validate_api_and_llm(self, data: Any) -> Any:
        """Validate the API and model."""
        api_config = data.get("api_config") or {}
        api = (
            data.get("api")
            if isinstance(data.get("api"), OpenAI)
            else LlmApiConfig(**api_config).create_openai_client()
        )
        if not api or not isinstance(api, OpenAI):
            raise ValueError("Unable to create OpenAI client.")

        data.update({"api": api})

        if not data.get("llm_name"):
            raise ValueError("No LLM API client or LLM name provided.")

        available_models: list[str] = [model.id for model in api.models.list()]
        if data.get("llm_name") not in available_models:
            raise ValueError(
                f"Model '{data.get('llm_name')}' is not available. "
                f"Available models: {', '.join(available_models)}"
            )
        return data
