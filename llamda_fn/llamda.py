from typing import Any, Callable, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from llamda_fn.llms.api_types import (
    LLMessage,
    LlToolCall,
    ToolResponse,
    LLCompletion,
    LLToolMessage,
    OaiToolParam,
)

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.llm_manager import LLManager
from llamda_fn.llms.exchange import Exchange
from llamda_fn.llms.type_transformers import ll_to_oai_message


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

        ll_completion: LLCompletion = self.api.chat_completion(
            messages=[
                ll_to_oai_message(msg, llm_name or self.api.llm_name)
                for msg in current_exchange
            ],
            tools=self.functions.get(tool_names),
        )

        current_exchange.append(ll_completion.message)
        if ll_completion.message.tool_calls:
            self._handle_tool_calls(ll_completion.message.tool_calls, current_exchange)

        return current_exchange[-1]

    def _handle_tool_calls(
        self, tool_calls: List[LlToolCall], exchange: Exchange
    ) -> None:
        execution_results: List[ToolResponse] = []  # Change list to List
        with ThreadPoolExecutor() as executor:
            futures: List[Future[ToolResponse]] = []  # Change list to List
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


__all__: List[str] = ["Llamda"]  # Change list to List
