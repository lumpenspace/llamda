<collection><title></title>

<document><path>llamda_fn/exchanges/exchange.py</path>
<content>
"""
Exchange class to represent a series of messages between a user and an assistant.
"""

from collections import UserList
from typing import List, Optional
from llamda_fn.llms import api_types as AT
from llamda_fn.llms.api_types import LLMessage


class Exchange(UserList[LLMessage]):
    """
    An exchange represents a series of messages between a user and an assistant.
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        messages: Optional[List[LLMessage]] = None,
    ) -> None:
        """
        Initialize the exchange.
        """
        super().__init__()
        if system_message:
            self.data.append(LLMessage(content=system_message, role="system"))
        if messages:
            self.data.extend(messages)

    def ask(self, content: str) -> None:
        """
        Add a user message to the exchange.
        """
        self.data.append(LLMessage(content=content, role="user"))

    def append(self, item: LLMessage) -> None:
        """
        Add a message to the exchange.
        """

        self.data.append(item)

    def get_context(self, n: int = 5) -> list[LLMessage]:
        """
        Get the last n messages as context.
        """
        return self.data[-n:]

    def __str__(self) -> str:
        """
        String representation of the exchange.
        """
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.data)

</content>
</document>

<document><path>llamda_fn/utils/logger.py</path>
<content>
"""
This module contains the console utilities for the penger package.
"""

from typing import Any
from rich.console import Console
from rich.live import Live
from rich.json import JSON

console = Console()
error_console = Console(stderr=True)


def live(shell: Console) -> Live:
    """
    Create a live console.
    """
    return Live(console=shell)


emojis = {
    "user": "🙎",
    "assistant": "🤖",
    "tool": "🔧",
    "system": "👽",
}


def log_message(role: str, message: Any, tool_call: bool = False) -> None:
    console.log(f"{emojis[role]}")
    console.log(JSON.from_data(message))


__all__ = ["console", "error_console", "live"]

</content>
</document>

<document><path>llamda_fn/examples/playground.py</path>
<content>
from typing import List, Tuple
from llamda_fn import Llamda
from llamda_fn.examples.functions.simple_function_aq import aq


ll = Llamda(
    system_message="""You are a cabalistic assistant who is eager to help users
    find weird numerical correspondences between strings.
    """
)


@ll.fy()
def aq_multiple(input_strings: List[str]) -> List[Tuple[str, int]]:
    """
    Calculate the Alphanumeric Quabala (AQ) value for multiple strings.

    This function calculates the AQ value for each string in the input list
    and returns a sorted list of tuples containing the original string and its AQ value.

    Args:
        input_strings (List[str]): A list of strings to calculate AQ values for.

    Returns:
        List[Tuple[str, int]]: A list of tuples (original_string, aq_value) sorted by AQ value.
    """
    return sorted([(s, aq(s)) for s in input_strings], key=lambda x: x[1])


ll.send_message("hello")

</content>
</document>

<document><path>llamda_fn/examples/functions/simple_function_aq.py</path>
<content>
"""Simple function to calculate the Alphanumeric Quabala (AQ) value of a string."""

import re


def aq(input_string: str) -> int:
    """
    Calculate the Alphanumeric Quabala (AQ) value of a string.

    This function calculates the sum of the numeric values of digits
    and the positional values of letters in the input string.

    Args:
        input_string (str): The input string to calculate the AQ for.

    Returns:
        int: The calculated AQ value.
    """
    input_string = re.sub(r"[^a-zA-Z0-9]", "", input_string.lower())
    digits = sum(int(char) for char in input_string if char.isdigit())
    letters = sum(ord(char) - 96 for char in input_string if char.isalpha())
    return digits + letters

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
    Sequence,
    Iterator,
)

from pydantic import BaseModel, ValidationError
from llamda_fn.llms.api_types import ToolCall, ToolResponse, OaiToolParam
from .llamda_classes import LlamdaFunction, LlamdaPydantic

R = TypeVar("R")
P = ParamSpec("P")


class LlamdaFunctions:
    """Creation and management of LLM tools"""

    def __init__(self) -> None:
        self._tools: Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]] = {}

    @property
    def tools(self) -> Dict[str, Union[LlamdaFunction[Any], LlamdaPydantic[Any]]]:
        """Returns the tools registered with the registry"""
        return self._tools

    def llamdafy(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Callable[P, R]], Union[LlamdaFunction[R], LlamdaPydantic[R]]]:
        """Decorator to mark a function as a tool and register it with the registry"""

        def decorator(
            func: Callable[P, R]
        ) -> Union[LlamdaFunction[R], LlamdaPydantic[R]]:
            func_name: str = name or func.__name__
            func_description: str = description or func.__doc__ or ""

            sig = signature(func)
            if len(sig.parameters) == 1:
                param = next(iter(sig.parameters.values()))
                if isclass(param.annotation) and issubclass(
                    param.annotation, BaseModel
                ):
                    llamda_func = LlamdaPydantic.create(
                        func_name, param.annotation, func_description, func
                    )
                    self._tools[func_name] = llamda_func
                    return llamda_func

            fields: Dict[str, tuple[type, Any]] = {
                param_name: (
                    param.annotation if param.annotation != Parameter.empty else Any,
                    param.default if param.default != Parameter.empty else ...,
                )
                for param_name, param in sig.parameters.items()
            }

            llamda_func: LlamdaFunction[R] = LlamdaFunction.create(
                func_name, fields, func_description, func
            )
            self._tools[func_name] = llamda_func
            return llamda_func

        return decorator

    def get(self, names: Optional[List[str]] = None) -> Sequence[OaiToolParam]:
        """Returns the tool spec for some or all of the functions in the registry"""
        if names is None:
            names = list(self._tools.keys())

        return [
            self._tools[name].to_tool_schema() for name in names if name in self._tools
        ]

    def execute_function(self, tool_call: ToolCall) -> ToolResponse:
        """Executes the function specified in the tool call with the required arguments"""
        try:
            if tool_call.name not in self._tools:
                raise KeyError(f"Function '{tool_call.name}' not found")

            parsed_args = json.loads(tool_call.arguments)
            result = self._tools[tool_call.name].run(**parsed_args)
        except KeyError as e:
            result: dict[str, str] = {"error": f"Error: {str(e)}"}
        except ValidationError as e:
            result = {"error": f"Error: Validation failed - {str(e)}"}
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}

        return ToolResponse(
            result,
            **tool_call.model_dump(),
        )

    def __getitem__(self, key: str) -> Union[LlamdaFunction[Any], LlamdaPydantic[Any]]:
        return self._tools[key]

    def __contains__(self, key: str) -> bool:
        return key in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

</content>
</document>

<document><path>llamda_fn/functions/process_fields.py</path>
<content>
from ast import List
from typing import Any, Dict, Union, get_args, get_origin
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
                    field_schema = {"type": "number", "nullable": True}
                elif non_none_type is int:
                    field_schema = {"type": "integer", "nullable": True}
                elif non_none_type is str:
                    field_schema = {"type": "string", "nullable": True}
                elif isinstance(non_none_type, type) and issubclass(
                    non_none_type, BaseModel
                ):
                    field_schema = {"type": "object", "nullable": True}

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

        # Remove 'title' field if present
        field_schema.pop("title", None)

        # Merge field_info with the generated schema
        if isinstance(field_info, dict):
            for key, value in field_info.items():
                if key not in field_schema or field_schema[key] is None:
                    field_schema[key] = value

        return field_type, field_schema
    except (SchemaError, ValidationError) as e:
        print(f"Error processing field: {e}")
        return Any, {"type": "any", "error": str(e)}


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

</content>
</document>

<document><path>llamda_fn/functions/llamda_classes.py</path>
<content>
from typing import Any, Callable, Dict, Generic, TypeVar
from pydantic import BaseModel, Field, create_model, ConfigDict

from llamda_fn.llms import OaiToolParam

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

    def to_tool_schema(self) -> OaiToolParam:
        """Get the JSON schema for the LlamdaPydantic."""
        schema = self.to_schema()
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

    def to_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for the LlamdaPydantic."""
        schema = self.model.model_json_schema()
        schema["title"] = self.name
        schema["description"] = self.description
        return schema

    def to_tool_schema(self) -> OaiToolParam:
        """Get the tool schema for the LlamdaPydantic."""
        schema: Dict[str, Any] = self.to_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["title"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema["properties"],
                    "required": schema.get("required", []),
                },
            },
        }

</content>
</document>

<document><path>llamda_fn/llamda.py</path>
<content>
"""
Llamda class to create, decorate, and run Llamda functions.
"""

from typing import Any, Callable, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from llamda_fn.functions.llamda_classes import R

from llamda_fn.llms.api_types import (
    LLMessage,
    LlToolCall,
    ToolResponse,
    LLToolMessage,
    OaiToolParam,
    LLCompletion,
)

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.llm_manager import LLManager
from llamda_fn.exchanges import Exchange
from llamda_fn.llms.type_transformers import ll_to_oai_message, oai_to_ll_completion


class Llamda:
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    def __init__(
        self,
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        self.api = LLManager(**kwargs)
        self.functions: LlamdaFunctions = LlamdaFunctions()
        self.exchange = Exchange(system_message=system_message)

    def fy(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self.functions.llamdafy(*args, **kwargs)

    @property
    def tools(self) -> Sequence[OaiToolParam]:
        """
        Get the tools available to the Llamda instance.
        """
        return self.functions.get()

    def run(
        self,
        tool_names: Optional[List[str]] = None,
        exchange: Optional[Exchange] = None,
        llm_name: Optional[str] = None,
    ) -> LLMessage:
        """
        Run the OpenAI API with the prepared data.
        """
        current_exchange: Exchange = exchange or self.exchange
        request = {
            "model": llm_name or self.api.llm_name,
            "messages": [ll_to_oai_message(msg) for msg in current_exchange],
        }
        tools: Sequence[OaiToolParam] = self.functions.get(tool_names)
        if tools:
            request["tools"] = tools

        oai_completion = self.api.chat_completion(**request)
        ll_completion: LLCompletion = oai_to_ll_completion(oai_completion)

        current_exchange.append(ll_completion.message)
        if ll_completion.message.tool_calls:
            self._handle_tool_calls(ll_completion.message.tool_calls, current_exchange)

        return current_exchange[-1]

    def _handle_tool_calls(
        self, tool_calls: List[LlToolCall], exchange: Exchange
    ) -> None:
        execution_results: list[ToolResponse] = []
        with ThreadPoolExecutor() as executor:
            futures: list[Future[ToolResponse]] = []
            for tool_call in tool_calls:
                futures.append(executor.submit(self._process_tool_call, tool_call))

            for future in as_completed(futures):
                result: ToolResponse = future.result()
                execution_results.append(result)
                exchange.append(LLToolMessage.from_execution(result))

        self.run(exchange=exchange)

    def _process_tool_call(self, tool_call: LlToolCall) -> ToolResponse:
        """
        Process a single tool call and return the result.
        """
        return self.functions.execute_function(tool_call=tool_call)

    def send_message(self, message: str) -> LLMessage:
        """
        Send a message and get a response.
        """
        self.exchange.ask(message)
        return self.run()


__all__: list[str] = ["Llamda"]

</content>
</document>

<document><path>llamda_fn/llms/api_types.py</path>
<content>
import uuid
from functools import cached_property
from typing import Any, Literal, Optional, Self, List

from openai.types.chat import ChatCompletion as OaiCompletion
from openai.types.chat import ChatCompletionMessageToolCall as OaiToolCall
from pydantic import BaseModel, Field, field_validator

Role = Literal["user", "system", "assistant", "tool"]


class LlToolCall(BaseModel):
    id: str
    name: str
    arguments: str

    @classmethod
    def from_oai_tool_call(cls, call: OaiToolCall) -> Self:
        return cls(
            id=call.id,
            name=call.function.name,
            arguments=call.function.arguments,
        )


class ToolResponse(BaseModel):
    id: str
    name: str
    arguments: str
    _result: str

    def __init__(self, result: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._result = result

    @cached_property
    def result(self) -> str:
        if isinstance(self._result, BaseModel):
            return self._result.model_dump_json()
        else:
            return self._result


class LLMessageMeta(BaseModel):
    choice: dict[str, Any] | None = Field(exclude=True)
    completion: dict[str, Any] | None = Field(exclude=True)


class LLMessage(BaseModel):
    id: Optional[str] = Field(..., exclude=True)
    role: Role
    content: str
    name: str | None = None
    tool_calls: List[LlToolCall] | None = None
    meta: LLMessageMeta | None = None

    @field_validator("id")
    @classmethod
    def add_id(cls, v: str | None) -> str:
        return v or str(uuid.uuid4())


class LLToolMessage(LLMessage):
    role: Role = "tool"

    @classmethod
    def from_execution(cls, execution: ToolResponse) -> Self:
        return cls(
            id=execution.id,
            name=execution.name,
            content=execution.result,
        )


class LLUserMessage(LLMessage):
    role: Role = "user"


class LLSystemMessage(LLMessage):
    role: Role = "system"


class LLAssistantMessage(LLMessage):
    role: Role = "assistant"


class LLCompletion(BaseModel):
    message: LLMessage
    meta: LLMessageMeta | None = None

    @classmethod
    def from_completion(cls, completion: OaiCompletion) -> Self:
        choice = completion.choices[0]
        message = choice.message
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                LlToolCall.from_oai_tool_call(tc) for tc in message.tool_calls
            ]

        return cls(
            message=LLMessage(
                id=completion.id,
                meta=LLMessageMeta(
                    choice=choice.model_dump(exclude={"message"}),
                    completion=completion.model_dump(exclude={"choices"}),
                ),
                role=message.role,
                content=message.content or "",
                tool_calls=tool_calls,
            )
        )

</content>
</document>

<document><path>llamda_fn/llms/llm_manager.py</path>
<content>
from typing import Any
from pydantic import Field, model_validator
from openai import OpenAI
from .api_types import LLCompletion, OaiCompletion
from .api import LlmApiConfig
from .type_transformers import oai_to_ll_completion


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

    def chat_completion(self, **kwargs: Any) -> LLCompletion:
        oai_completion: OaiCompletion = super().chat.completions.create(**kwargs)
        return oai_to_ll_completion(oai_completion)

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

</content>
</document>

<document><path>llamda_fn/llms/api.py</path>
<content>
"""Module to handle the LLM APIs."""

from os import environ
from typing import Any, Optional
import dotenv

from pydantic import BaseModel, Field, field_validator
from openai import OpenAI

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


__all__: list[str] = [
    "LlmApiConfig",
]

</content>
</document>

<document><path>llamda_fn/llms/type_transformers.py</path>
<content>
from typing import Any
from llamda_fn.llms.api_types import (
    LLMessage,
    LLCompletion,
    LlToolCall,
    OaiResponseMessage,
    OaiCompletion,
    OaiToolCall,
)


def ll_to_oai_message(message: LLMessage) -> OaiResponseMessage:
    oai_message = {
        "role": message.role,
        "content": message.content,
    }
    if message.name:
        oai_message["name"] = message.name
    if message.tool_calls:
        oai_message["tool_calls"] = [tc.to_oai_tool_call() for tc in message.tool_calls]
    return OaiResponseMessage(**oai_message)


def oai_to_ll_completion(completion: OaiCompletion) -> LLCompletion:
    return LLCompletion.from_completion(completion)

</content>
</document>

</collection>