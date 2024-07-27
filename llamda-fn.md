<collection><title></title>

<document><path>llamda_fn/exchanges/exchange.py</path>
<content>
from collections import UserList
from typing import List, Literal

from openai.types.chat import ChatCompletionMessageParam

from llamda_fn.exchanges.messages import to_message


class Exchange(UserList[ChatCompletionMessageParam]):
    """
    An exchange represents a series of messages between a user and an assistant.
    """

    data: List[ChatCompletionMessageParam]

    def __init__(
        self,
        system_message: str | None = None,
        messages: List[ChatCompletionMessageParam] | None = None,
    ):
        """
        Initialize the exchange.
        """
        if system_message:
            messages = [to_message(system_message, "system")] + (messages or [])
        self.data = messages or []

    def append(
        self,
        item: str | ChatCompletionMessageParam,
        role: Literal["user", "system", "assistant"] = "user",
    ) -> None:
        """
        Push a message to the exchange.
        """
        if isinstance(item, str):
            self.data.append(to_message(item, role))
        else:
            self.data.append(item)

    def clear(self) -> None:
        """
        Clear the exchange.
        """
        self.data.clear()

</content>
</document>

<document><path>llamda_fn/exchanges/messages.py</path>
<content>
"""
This module contains functions related to verbal/conversational messages.
"""

from typing import Any, Optional, Literal

from llamda_fn.utils.api import (
    UserMessage,
    SystemMessage,
    AssistantMessage,
    MessageParam,
)


def to_message(
    text: str,
    role: Literal["user", "system", "assistant"],
    name: Optional[str] = None,
) -> MessageParam:
    """
    Create a message.
    """
    base_dict: dict[str, Any] = {"content": text}
    if name is not None:
        base_dict["name"] = name
    match role:
        case "user":
            message = UserMessage(**base_dict, role="user")
        case "system":
            message = SystemMessage(**base_dict, role="system")
        case "assistant":
            message = AssistantMessage(**base_dict, role="assistant")
    return message

</content>
</document>

<document><path>llamda_fn/utils/llamda_validator.py</path>
<content>
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

</content>
</document>

<document><path>llamda_fn/utils/logger.py</path>
<content>
"""
This module contains the console utilities for the penger package.
"""

from rich.console import Console
from rich.live import Live

console = Console()
error_console = Console(stderr=True)


def live(shell: Console) -> Live:
    """
    Create a live console.
    """
    return Live(console=shell)


__all__ = ["console", "error_console", "live"]

</content>
</document>

<document><path>llamda_fn/utils/api.py</path>
<content>
"""Module to handle the LLM APIs."""

from os import environ
from typing import Any, Optional
import dotenv

from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageToolCall as ToolCall,
    ChatCompletionToolMessageParam as ToolMessage,
    ChatCompletionToolParam as ToolParam,
    ChatCompletion as ChatCompletion,
    ChatCompletionMessage as Message,
    ChatCompletionAssistantMessageParam as AssistantMessage,
    ChatCompletionSystemMessageParam as SystemMessage,
    ChatCompletionUserMessageParam as UserMessage,
    ChatCompletionMessageParam as ChatMessage,
    ChatCompletionMessageParam as MessageParam,
)


dotenv.load_dotenv()


class LlmApiConfig(BaseModel):
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

    def create_openai_client(self) -> OpenAI:
        """
        Create and return an OpenAI client with the configured settings.
        """
        config = {k: v for k, v in self.model_dump().items() if v is not None}
        return OpenAI(**config)


__all__ = [
    "LlmApiConfig",
    "ToolParam",
    "ToolCall",
    "ToolMessage",
    "AssistantMessage",
    "SystemMessage",
    "UserMessage",
    "ChatMessage",
    "ChatCompletion",
    "MessageParam",
    "Message",
]

</content>
</document>

<document><path>llamda_fn/examples/playground.py</path>
<content>

</content>
</document>

<document><path>llamda_fn/examples/functions/simple_function_aq.py</path>
<content>
"""
Alphanumeric Quabala.

A simple function that takes a string and returns the AQ of the string.
"""

import re


def aq(input_string: str) -> str:
    """
    Calculate the AQ of a string.

    It is used to find correspondences between words, phrases, etc.
    """
    input_string = re.sub(r"[^a-zA-Z0-9]", "", input_string.lower())
    digits = sum(int(char) for char in input_string)
    letters = sum(ord(char) - 96 for char in input_string)
    return str(digits + letters)


def aq_multiple(input_strings: list[str]) -> list[tuple[str, str]]:
    """Calculates the alphanumeric cabala value of a list of strings and return tuples of
    (string, aq_value) sorted by the cabala value."""
    return sorted([(s, aq(s)) for s in input_strings], key=lambda x: x[1])

</content>
</document>

<document><path>llamda_fn/functions/llamda_functions.py</path>
<content>
import json

from inspect import Parameter, isclass, signature
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    ParamSpec,
    Union,
)
from git import Sequence
from pydantic import BaseModel, ValidationError
from llamda_fn.utils.api import (
    ToolCall,
    ToolParam,
    ToolMessage,
)
from .llamda_classes import LlamdaFunction, LlamdaPydantic

R = TypeVar("R")
P = ParamSpec("P")


class LlamdaFunctions:
    """
    Main class, produces a decorator for creating Llamda functions and manages their execution.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]] = {}

    def llamdafy(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[P, R]], Union[LlamdaFunction[R], LlamdaPydantic[R]]]:
        """
        Decorator for creating Llamda functions.
        """

        def decorator(
            func: Callable[P, R]
        ) -> Union[LlamdaFunction[R], LlamdaPydantic[R]]:
            func_name: str = name or func.__name__
            func_description: str = description or func.__doc__ or ""

            # Check if the function is expecting a Pydantic model
            sig = signature(func)
            if len(sig.parameters) == 1:
                param = next(iter(sig.parameters.values()))
                if isclass(param.annotation) and issubclass(
                    param.annotation, BaseModel
                ):
                    llamda_func = LlamdaPydantic.create(
                        func_name, param.annotation, func_description, func
                    )
                    self._tools.update({func_name: llamda_func})
                    print(f"{func_name=}")
                    return llamda_func

            # If not a Pydantic model, treat it as a regular function
            fields: Dict[str, tuple[type, Any]] = {}
            for param_name, param in sig.parameters.items():
                field_type = (
                    param.annotation if param.annotation != Parameter.empty else Any
                )
                field_default = (
                    param.default if param.default != Parameter.empty else ...
                )
                fields[param_name] = (field_type, field_default)

            llamda_func = LlamdaFunction.create(
                func_name, fields, func_description, func
            )
            print(f"{func_name=}")
            self._tools.update({func_name: llamda_func})
            return llamda_func

        return decorator

    @property
    def tools(self) -> Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]]:
        """
        Get the tools.
        """
        return self._tools

    def get(self, names: Optional[List[str]] = None) -> Sequence[ToolParam]:
        """
        Prepare the tools for the OpenAI API.
        """
        tools = []
        if names is None:
            names = list(self._tools.keys())

        for name in names:
            if name in self._tools:
                tool_schema: Dict[str, Any] = self._tools[name].to_schema()
                tools.append(tool_schema)
        return tools

    def execute_function(self, tool_call: ToolCall) -> ToolMessage:
        """
        Execute a function and return the result or error message.
        """
        try:
            parsed_args = json.loads(tool_call.function.arguments)
            result = self._tools[tool_call.function.name].run(**parsed_args)
        except ValidationError as e:
            result: dict[str, str] = {"error": f"Error: Validation failed - {str(e)}"}
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}
        return ToolMessage(
            content=json.dumps(result),
            role="tool",
            tool_call_id=tool_call.id,
        )

</content>
</document>

<document><path>llamda_fn/functions/process_fields.py</path>
<content>
from ast import List
from typing import Any, Dict, Optional, Union, get_args, get_origin
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import SchemaError

JsonDict = Dict[str, Any]


def process_field(
    field_type: Any, field_info: Union[JsonDict, FieldInfo]
) -> tuple[Any, JsonDict]:
    """
    Process a field type and info, using Pydantic's model_json_schema for schema generation.
    """
    try:
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Handle nested Pydantic models
            nested_schema = field_type.model_json_schema()
            field_schema = {
                "type": "object",
                "properties": nested_schema.get("properties", {}),
                "required": nested_schema.get("required", []),
            }
        else:
            # Create a temporary model with the field
            if isinstance(field_info, FieldInfo):
                temp_field = field_info
            else:
                temp_field = Field(**field_info)

            TempModel = create_model("TempModel", field=(field_type, temp_field))

            # Get the JSON schema for the entire model
            full_schema = TempModel.model_json_schema()

            # Extract the schema for our specific field
            field_schema = full_schema["properties"]["field"]

        # Handle Optional types
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            if type(None) in args:
                # This is an Optional type
                non_none_type = next(arg for arg in args if arg is not type(None))
                if non_none_type is float:
                    field_schema["type"] = "number"
                elif non_none_type is int:
                    field_schema["type"] = "integer"
                elif non_none_type is str:
                    field_schema["type"] = "string"
                elif isinstance(non_none_type, type) and issubclass(
                    non_none_type, BaseModel
                ):
                    field_schema["type"] = "object"
                field_schema["nullable"] = True

        # Ensure 'type' is always set
        if "type" not in field_schema:
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                field_schema["type"] = "object"
            elif field_type is int:
                field_schema["type"] = "integer"
            elif field_type is float:
                field_schema["type"] = "number"
            elif field_type is str:
                field_schema["type"] = "string"
            elif field_type is bool:
                field_schema["type"] = "boolean"
            elif field_type is list or field_type is List:
                field_schema["type"] = "array"
            elif field_type is dict or field_type is Dict:
                field_schema["type"] = "object"
            else:
                field_schema["type"] = "any"

        # Merge field_info with the generated schema
        if isinstance(field_info, dict):
            for key, value in field_info.items():
                if key not in field_schema or field_schema[key] is None:
                    field_schema[key] = value

        return field_type, field_schema
    except (SchemaError, ValidationError) as e:
        print(f"Error processing field: {e}")
        return Any, {"type": "any", "error": str(e)}

        # If schema generation fails, return Any type a


def process_fields(fields: Dict[str, Any]) -> Dict[str, tuple[Any, JsonDict]]:
    """
    Process all fields in a model, using Pydantic for complex types.
    """
    processed_fields = {}
    for field_name, field_value in fields.items():
        if isinstance(field_value, FieldInfo):
            field_type = field_value.annotation
            field_info = field_value
        elif isinstance(field_value, tuple):
            field_type, field_info = field_value
        else:
            raise ValueError(
                f"Unexpected field value type for {field_name}: {type(field_value)}"
            )

        processed_type, processed_info = process_field(field_type, field_info)

        # Ensure 'type' is set for nested Pydantic models
        if isinstance(processed_type, type) and issubclass(processed_type, BaseModel):
            processed_info["type"] = "object"

        processed_fields[field_name] = (processed_type, processed_info)

    return processed_fields


def create_model_from_fields(
    name: str, fields: Dict[str, tuple[Any, JsonDict]]
) -> type[BaseModel]:
    """
    Create a Pydantic model from a dictionary of fields.
    """
    model_fields = {}
    for field_name, (field_type, field_info) in fields.items():
        default = field_info.get("default", ...)
        if default is None:
            field_type = Optional[field_type]
            field_info["default"] = None
        model_fields[field_name] = (field_type, Field(**field_info))

    return create_model(name, **model_fields)

</content>
</document>

<document><path>llamda_fn/functions/llamda_classes.py</path>
<content>
from typing import Any, Callable, Dict, Generic, TypeVar
from pydantic import BaseModel, Field, create_model, ConfigDict

from llamda_fn.functions.process_fields import JsonDict

R = TypeVar("R")


class LlamdaBase(BaseModel, Generic[R]):
    """The base class for Llamda functions."""

    name: str
    description: str
    call_func: Callable[..., Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self, **kwargs: Any) -> Any:
        """Run the Llamda function with the given parameters."""
        raise NotImplementedError

    def to_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the Llamda function."""
        raise NotImplementedError


class LlamdaFunction(LlamdaBase[R]):
    """A Llamda function that uses a simple function model as the input."""

    parameter_model: type[BaseModel]

    @classmethod
    def create(
        cls,
        name: str,
        fields: Dict[str, tuple[type, Any]],
        description: str,
        call_func: Callable[..., Any],
    ) -> "LlamdaFunction[R]":
        """Create a new LlamdaFunction from a function."""
        model_fields = {}
        for field_name, (field_type, field_default) in fields.items():
            if field_default is ...:
                model_fields[field_name] = (field_type, Field(...))
            else:
                model_fields[field_name] = (field_type, Field(default=field_default))

        parameter_model = create_model(f"{name}Parameters", **model_fields)

        return cls(
            name=name,
            description=description,
            parameter_model=parameter_model,
            call_func=call_func,
        )

    def run(self, **kwargs: Any) -> Any:
        """Run the LlamdaFunction with the given parameters."""
        validated_params = self.parameter_model(**kwargs)
        return self.call_func(**validated_params.model_dump())

    def to_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the LlamdaFunction."""
        schema = self.parameter_model.model_json_schema()
        schema["title"] = self.name
        schema["description"] = self.description
        return schema


class LlamdaPydantic(LlamdaBase[R]):
    """A Llamda function that uses a Pydantic model as the input."""

    model: type[BaseModel]

    @classmethod
    def create(
        cls,
        name: str,
        model: type[BaseModel],
        description: str,
        call_func: Callable[..., Any],
    ) -> "LlamdaPydantic[R]":
        """Create a new LlamdaPydantic from a Pydantic model."""
        return cls(
            name=name,
            description=description,
            model=model,
            call_func=call_func,
        )

    def run(self, **kwargs: Any) -> Any:
        """Run the LlamdaPydantic with the given parameters."""
        validated_params = self.model(**kwargs)
        return self.call_func(validated_params)

    def to_schema(self) -> JsonDict:
        """Get the JSON schema for the LlamdaPydantic."""
        schema = self.model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["title"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema["required"],
                },
            },
        }

</content>
</document>

<document><path>llamda_fn/llamda.py</path>
<content>
from typing import Any, Callable, List, Optional, Sequence

from openai import OpenAI

from llamda_fn.utils.api import (
    ChatMessage,
    ToolCall,
    ToolParam,
    ChatCompletion,
    ToolMessage,
    Message,
)


from llamda_fn.functions import LlamdaFunctions
from llamda_fn.utils import LlamdaValidator
from llamda_fn.exchanges import Exchange


class Llamda:
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    api: OpenAI
    llm_name: str
    retry: int
    llamda_functions: LlamdaFunctions
    exchange: Exchange

    def __init__(
        self,
        api: Optional[OpenAI] = None,
        api_config: Optional[dict[str, Any]] = None,
        llm_name: str = "gpt-4o-mini",
        system_message: Optional[str] = None,
    ):
        validator = LlamdaValidator(
            api=api, api_config=api_config or {}, llm_name=llm_name
        )
        if not validator.api:
            raise ValueError("API is not set.")

        self.api: OpenAI = validator.api or OpenAI()
        self.llm_name: str = validator.llm_name
        self.functions: LlamdaFunctions = LlamdaFunctions()
        self.exchange = Exchange(system_message=system_message)

    def llamdafy(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self.llamda_functions.llamdafy(*args, **kwargs)

    @property
    def tools(self) -> Sequence[ToolParam]:
        """
        Get the tools available to the Llamda instance.
        """
        return self.functions.get()

    def run(
        self,
        tool_names: List[str] | None = None,
        exchange: Exchange | None = None,
        llm_name: str | None = None,
    ) -> ChatMessage | ToolMessage:
        """
        Run the OpenAI API with the prepared data.
        """
        request: dict[str, Any] = {
            "model": llm_name or self.llm_name,
            "messages": exchange or self.exchange,
        }

        tools: Sequence[ToolParam] = self.functions.get(tool_names)
        if tools and len(tools) > 0:
            request["tools"] = tools

        print(f"{request=}")
        response: ChatCompletion = self.api.chat.completions.create(**request)
        message: Message = response.choices[0].message
        if message.tool_calls:
            self.execute_calls(message.tool_calls)
        else:
            self.exchange.append(message.content or "No response", role="assistant")
        return self.exchange[-1]

    def execute_calls(self, tool_calls: List[ToolCall]) -> None:
        """
        Execute the tool calls.
        """
        for tool_call in tool_calls:
            self.exchange.append(
                self.llamda_functions.execute_function(tool_call=tool_call)
            )
        self.run()


__all__: list[str] = ["Llamda"]

</content>
</document>

</collection>