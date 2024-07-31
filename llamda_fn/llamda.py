from typing import Any, Callable, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from llamda_fn.llms.ll_tool import LLToolResponse, LLToolCall
from llamda_fn.llms.ll_message import LLMessage
from llamda_fn.utils.logger import LOG

from llamda_fn.llms.oai_api_types import OaiToolSpec


from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.ll_manager import LLManager
from llamda_fn.llms.ll_exchange import LLExchange


class Llamda:
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    def __init__(
        self,
        system: Optional[str | LLMessage] = None,
        max_consecutive_tool_calls: int = 0,
        llm_name: str = "gpt-4o-mini",
        **kwargs: Any,
    ):
        self.max_consecutive_tool_calls: int = max_consecutive_tool_calls

        self.api = LLManager(llm_name=llm_name, **kwargs)
        self.functions: LlamdaFunctions = LlamdaFunctions()
        self.exchange = LLExchange(system=system)

    def fy(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self.functions.llamdafy(*args, **kwargs)

    @property
    def tools(self) -> Sequence[OaiToolSpec]:
        """
        Get the tools available to the Llamda instance.
        """
        return self.functions.spec

    def run(
        self,
        tool_names: Optional[List[str]] = None,
        exchange: Optional[LLExchange] = None,
        llm_name: Optional[str] = None,
    ) -> LLMessage:
        """
        Run the OpenAI API with the prepared data.
        """
        current_exchange: LLExchange = exchange or self.exchange

        with LOG.live_logging():
            ll_completion: LLMessage = self.api.query(
                messages=current_exchange,
                llm_name=llm_name or self.api.llm_name,
                tools=self.functions.tools if self.functions.tools else None,
            )
            current_exchange.append(ll_completion)
            if ll_completion.tool_calls:
                self._handle_tool_calls(ll_completion.tool_calls)

        return current_exchange[-1]

    def _handle_tool_calls(self, tool_calls: Sequence[LLToolCall]) -> None:
        LOG.tool_calls(tool_calls)
        with ThreadPoolExecutor() as executor:
            futures: List[Future[LLToolResponse]] = [
                executor.submit(self._process_tool_call, tool_call)
                for tool_call in tool_calls
            ]
            for future in as_completed(futures):
                result: LLToolResponse = future.result()
                self.exchange.append(LLMessage.from_tool_response(result))
        self.run()

    def _process_tool_call(self, tool_call: LLToolCall) -> LLToolResponse:
        """
        Process a single tool call and return the result.
        """
        result: LLToolResponse = self.functions.execute_function(tool_call=tool_call)
        return result

    def __call__(self, text: str) -> LLMessage:
        """
        Send a message and get a response.
        """
        self.exchange.ask(text)
        return self.run()


__all__: List[str] = ["Llamda"]  # Change list to List
