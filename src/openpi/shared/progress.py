"""Keep inherited tqdm displays readable in terminals and task log files."""

import sys


def sync_pbar_color(pbar):
    if hasattr(pbar, "colour"):
        pbar.colour = "cyan" if sys.stderr.isatty() else None
