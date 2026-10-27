"""Small log-formatting helpers used by the inherited pistar scripts."""

import sys


def _format(message, color):
    text = str(message)
    return f"\033[{color}m{text}\033[0m" if sys.stderr.isatty() else text


def info(message):
    return _format(message, 36)


def ok(message):
    return _format(message, 32)


def warn(message):
    return _format(message, 33)


def error(message):
    return _format(message, 31)
