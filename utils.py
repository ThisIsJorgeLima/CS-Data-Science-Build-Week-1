"""
Import Statements:
"""

from os import name, system


def clear_terminal():
    if name == "nt":
        # Windows
        system("cls")
    else:
        # Linux or Mac
        system("clear")
