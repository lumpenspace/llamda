from llamda_fn import Llamda
from llamda_fn.examples import aq_multiple

global ll


def doit():
    """
    Create a Llamda instance with a system message and a function.
    """

    ll = Llamda(
        system_message="You are a cabalistic assistant who is eager to help user find weird numerical correspondences"
    )

    ll.fy()(aq_multiple)


doit()
