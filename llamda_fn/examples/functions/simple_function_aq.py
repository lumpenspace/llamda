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
