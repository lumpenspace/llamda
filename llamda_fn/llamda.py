from typing import Any, Callable, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from llamda_fn.llms import ll_tool
from llamda_fn.llms.ll_tool import LLToolResponse, LlToolCall
from llamda_fn.llms.ll_message import LLMessage
from llamda_fn.utils.logger import LOG

from llamda_fn.llms.oai_api_types import OaiToolSpec, OaiToolCall


from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.ll_manager import LLManager
from llamda_fn.llms.ll_exchange import LLExchange


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
        return self.functions.get()

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

        ll_completion: LLMessage = self.api.query(
            messages=current_exchange,
            llm_name=llm_name or self.api.llm_name,
            tools=self.functions.get(tool_names),
        )
        LOG.msg(ll_completion)
        current_exchange.append(ll_completion)
        if ll_completion.tool_calls:
            self._handle_tool_calls(ll_completion.tool_calls)

        return current_exchange[-1]

    def _handle_tool_calls(self, tool_calls: Sequence[LlToolCall]) -> None:

        tool_log = LOG.tools(tool_calls)
        with ThreadPoolExecutor() as executor:
            futures: List[Future[LLToolResponse]] = [
                executor.submit(self._process_tool_call, tool_call, tool_log)
                for tool_call in tool_calls
            ]
            for future in as_completed(futures):
                result: LLToolResponse = future.result()
                self.exchange.append(result.oai)
        self.run()

    def _process_tool_call(
        self,
        tool_call: LlToolCall,
        tool_log: Callable[[LlToolCall, LLToolResponse], None],
    ) -> LLToolResponse:
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
