"""
Llamda class to create, decorate, and run Llamda functions.
"""

from typing import Any, Callable, List, Optional, Sequence
from llamda_fn.llms.ll_tool import LLToolResponse, LLToolCall
from llamda_fn.llms.ll_message import LLMessage


from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.ll_manager import LLManager
from llamda_fn.llms.ll_exchange import LLExchange
from llamda_fn.llms.oai_api_types import OaiRole


class Llamda:
    """
    Llamda class to create, decorate, and run Llamda functions.
    """

    exchange: LLExchange

    def __init__(
        self,
        system: Optional[str | LLMessage] = None,
        max_consecutive_tool_calls: int = 0,
        llm_name: str = "gpt-4o-mini",
        **kwargs: Any,
    ):
        super().__init__()

        self.max_consecutive_tool_calls: int = max_consecutive_tool_calls
        self.api = LLManager(llm_name=llm_name, **kwargs)
        self._functions: LlamdaFunctions = LlamdaFunctions()
        self.exchange = LLExchange(system=system)

    def fy(
        self,
        *args: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """
        Decorator method to create a Llamda function.
        """
        return self._functions.llamdafy(name=name, description=description, **kwargs)

    @property
    def tools(self) -> LlamdaFunctions:
        """
        Get the tools available to the Llamda instance.
        """
        return self._functions

    def _handle_tool_calls(self, tool_calls: Sequence[LLToolCall]) -> None:
        for tool_call in tool_calls:
            response = self._process_tool_call(tool_call)
            self.exchange.append(LLMessage.from_tool_response(response))

    def _process_tool_call(self, tool_call: LLToolCall) -> LLToolResponse:
        """
        Process a single tool call and return the result.
        """
        try:
            result: LLToolResponse = self._functions.execute_function(
                tool_call=tool_call
            )
            return result
        except Exception as e:
            raise e

    def run(
        self,
        llm_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMessage:
        """
        Run the OpenAI API with the prepared data.
        """
        current_exchange: LLExchange = self.exchange

        ll_completion: LLMessage = self.api.query(
            messages=current_exchange,
            llm_name=llm_name or self.api.llm_name,
            tools=self._functions.spec if self._functions.spec else None,
        )
        current_exchange.append(ll_completion)
        if ll_completion.tool_calls:
            self._handle_tool_calls(ll_completion.tool_calls)
            return self.run(llm_name=llm_name, **kwargs)
        else:
            return ll_completion  # Return the final message without tool calls

    def ask(
        self, text: str, *args: Any, role: OaiRole = "user", **kwargs: Any
    ) -> LLMessage:
        """ """
        self.exchange.append(LLMessage(role=role, content=text))
        return self.run(**kwargs)


__all__: List[str] = ["Llamda"]
