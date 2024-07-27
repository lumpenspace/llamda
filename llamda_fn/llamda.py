import json
from typing import Any, Callable, List, Optional, Sequence

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field, model_validator

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.utils import LlmApiConfig


class Llamda(BaseModel):
    """
    Llamda class to create, decorate and run Llamda functions.
    """

    llamda_functions: LlamdaFunctions = Field(default_factory=LlamdaFunctions)
    llm_api: OpenAI = Field(...)
    llm_api_config: dict[str, Any] = Field(default_factory=dict)
    retry: int = Field(default=3)
    llm_name: str = Field(default="gpt-4-0613")

    @model_validator(mode="before")
    def model_validator(self, values: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the model and create the OpenAI client if needed.
        """
        llm_api = values.get("llm_api", None)
        if not llm_api:
            llm_api: OpenAI = LlmApiConfig(**llm_api).create_openai_client()

        llm_name: Any | None = values.get("llm_name")
        if isinstance(values["llm_api"], OpenAI) and llm_name:
            available_models = list(values["llm_api"].models.list())
            model_ids = [model.id for model in available_models]
            if llm_name not in model_ids:
                raise ValueError(
                    f"Model '{llm_name}' is not available. "
                    f"Available models: {', '.join(model_ids)}"
                )

        return {
            "llamda_functions": LlamdaFunctions(),
            "llm_api": llm_api,
            "retry": values.get("retry"),
            "llm_name": values.get("llm_name"),
        }

    class Config:
        """
        Config for the Llamda class.
        """

        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.llamda_functions = self.llamda_functions or LlamdaFunctions()
        self.llm_api = self.llm_api

    def llamdafy(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self.llamda_functions.llamdafy(*args, **kwargs)

    def run(
        self,
        messages: list[ChatCompletionMessageParam],
        function_names: Optional[List[str]] = None,
    ) -> str:
        """
        Run the OpenAI API with the prepared data.
        """
        tools: Sequence[ChatCompletionToolParam] = self.llamda_functions.prepare_tools(
            function_names
        )

        for _ in range(self.retry + 1):
            response: ChatCompletion = self.llm_api.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                tools=tools,
            )
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
