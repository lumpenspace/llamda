from typing import Any, Callable, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from llamda_fn.utils.logger import logger

from llamda_fn.llms.api_types import (
    LLCompletion,
    LLMessage,
    LlToolCall,
    ToolResponse,
    OaiToolParam,
)

from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.llm_manager import LLManager
from llamda_fn.llms.exchange import Exchange


class Llamda:
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    def __init__(
        self,
        system: Optional[str] = None,
        **kwargs: Any,
    ):
        self.api = LLManager(**kwargs)
        self.functions: LlamdaFunctions = LlamdaFunctions()
        self.exchange = Exchange(system=system)

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
            messages=current_exchange,
            llm_name=llm_name or self.api.llm_name,
            tools=self.functions.get(tool_names),
        )
        logger.msg(ll_completion.message)
        current_exchange.append(ll_completion.message)
        if ll_completion.message.tool_calls:
            self._handle_tool_calls(ll_completion.message.tool_calls)

        return current_exchange[-1]

    def _handle_tool_calls(self, tool_calls: List[LlToolCall]) -> None:

        tool_log = logger.tools(tool_calls)
        with ThreadPoolExecutor() as executor:
            futures: List[Future[ToolResponse]] = [
                executor.submit(self._process_tool_call, tool_call, tool_log)
                for tool_call in tool_calls
            ]
            for future in as_completed(futures):
                result: ToolResponse = future.result()
                self.exchange.append(LLMessage.from_execution(result))
        self.run()

    def _process_tool_call(
        self,
        tool_call: LlToolCall,
        tool_log: Callable[[LlToolCall, ToolResponse], None],
    ) -> ToolResponse:
        """
        Process a single tool call and return the result.
        """
        result = self.functions.execute_function(tool_call=tool_call)
        tool_log(tool_call, result)
        return result

    def __call__(self, text: str) -> LLMessage:
        """
        Send a message and get a response.
        """
        self.exchange.ask(text)
        return self.run()


__all__: List[str] = ["Llamda"]  # Change list to List
