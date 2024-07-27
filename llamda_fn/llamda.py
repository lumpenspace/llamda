import json
from typing import Any, Callable, List, Optional, Self, Sequence, TypedDict, NotRequired

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.model import Model as OpenAIModel
from pydantic import BaseModel, Field, model_validator

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.utils import LlmApiConfig


Request = TypedDict(
    "Request",
    {
        "model": str,
        "messages": List[ChatCompletionMessageParam],
        "tools": NotRequired[List[ChatCompletionToolParam]],
    },
)


class Llamda(BaseModel):
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    llamda_functions: LlamdaFunctions = Field(..., default_factory=LlamdaFunctions)
    api: Optional[OpenAI]
    api_config: dict[str, Any] = Field(..., default_factory=dict)
    retry: int = Field(default=3)
    llm_name: str = Field(default="gpt-4-0613")

    @model_validator(mode="before")
    @classmethod
    def class_validator(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the model and create the OpenAI client if needed.
        """
        api = data.get("api")
        api_config = data.get("api_config") or {}
        print(api_config, api)
        if not api:
            api = LlmApiConfig(**api_config).create_openai_client()
            if not api:
                raise ValueError("Unable to create OpenAI client.")

        llm_name: str = data.get("llm_name") or "gpt-4o"
        if llm_name and api:
            available_models: list[OpenAIModel] = list(api.models.list())
            model_ids = [model.id for model in available_models]
            if llm_name not in model_ids:
                raise ValueError(
                    f"Model '{llm_name}' is not available. "
                    f"Available models: {', '.join(model_ids)}"
                )
        else:
            raise ValueError("No LLM API client or LLM name provided.")

        data["api"] = api
        return data

    @model_validator(mode="after")
    def instance_validator(self) -> Self:
        """
        Validate the model and create the OpenAI client if needed.
        """
        if not self.api:
            raise ValueError("No LLM API client provided.")
        if not self.llamda_functions:
            self.llamda_functions = LlamdaFunctions()
        return self

    class Config:
        """
        Config for the Llamda class.
        """

        arbitrary_types_allowed = True

    def llamdafy(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self.llamda_functions.llamdafy(*args, **kwargs)

    def run(
        self,
        messages: List[ChatCompletionMessageParam],
        function_names: Optional[List[str]] = None,
    ) -> str:
        """
        Run the OpenAI API with the prepared data.
        """
        tools: Sequence[ChatCompletionToolParam] = self.llamda_functions.prepare_tools(
            function_names
        )

        request: dict[str, Any] = {
            "model": self.llm_name,
            "messages": messages,
        }
        if tools and len(tools) > 0:
            request["tools"] = tools

        for _ in range(self.retry + 1):
            response: ChatCompletion = self.api.chat.completions.create(**request)
            assistant_message: ChatCompletionMessage = response.choices[0].message
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    function_result = self.llamda_functions.execute_function(
                        tool_call.function.name, tool_call.function.arguments
                    )
                    messages.append(
                        ChatCompletionFunctionMessageParam(
                            role="function",
                            content=json.dumps(function_result),
                            name=tool_call.function.name,
                        )
                    )
            else:
                return assistant_message.content or "No response"

        return "Max retries reached. Unable to complete the request."


__all__ = ["Llamda"]
