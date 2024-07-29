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
