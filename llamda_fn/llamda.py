"""
Llamda class to create, decorate, and run Llamda functions.
"""

from typing import Any, Callable, List, Optional, Sequence
from llamda_fn.llms.ll_tool import LLToolResponse, LLToolCall
from llamda_fn.llms.ll_message import LLMessage

from llamda_fn.llms.oai_api_types import OaiToolSpec


from llamda_fn.functions import LlamdaFunctions
from llamda_fn.llms.ll_manager import LLManager
from llamda_fn.llms.ll_exchange import LLExchange
from llamda_fn.llogos import LlogosMixin


class Llamda(LlogosMixin):
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
        super().__init__()

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

    def _handle_tool_calls(self, tool_calls: Sequence[LLToolCall]) -> None:
        for tool_call in tool_calls:
            result: LLToolResponse = self._process_tool_call(tool_call)
            self.exchange.append(LLMessage.from_tool_response(response=result))
            self.msg.update_tool(
                result.tool_call_id, "Success" if result.success else "Error"
            )

    def _process_tool_call(self, tool_call: LLToolCall) -> LLToolResponse:
        """
        Process a single tool call and return the result.
        """
        try:
            result: LLToolResponse = self.functions.execute_function(
                tool_call=tool_call
            )
            return result
        except Exception as e:
            raise e

    def run(
        self,
        exchange: Optional[LLExchange] = None,
        llm_name: Optional[str] = None,
    ) -> LLMessage:
        """
        Run the OpenAI API with the prepared data.
        """
        current_exchange: LLExchange = exchange or self.exchange

        while True:
            ll_completion: LLMessage = self.api.query(
                messages=current_exchange,
                llm_name=llm_name or self.api.llm_name,
                tools=self.functions.spec if self.functions.spec else None,
            )
            current_exchange.append(ll_completion)
            self.msg(ll_completion)
            if ll_completion.tool_calls:
                self._handle_tool_calls(ll_completion.tool_calls)
            else:
                return ll_completion  # Return the final message without tool calls

    def __call__(self, user_input: str) -> LLMessage:
        # Add user input to the exchange
        self.exchange.append(LLMessage(role="user", content=user_input))

        # Run the LLM
        result = self.run()

        return result


__all__: List[str] = ["Llamda"]
