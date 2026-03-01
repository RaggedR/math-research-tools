"""Shared utility functions for the kg package."""

import re


def slugify(name):
    """Convert a name to a filesystem-safe slug.

    Lowercases, strips non-word characters, replaces whitespace with hyphens.
    Truncates to 80 characters.
    """
    s = name.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s]+', '-', s)
    s = re.sub(r'-+', '-', s)
    return s.strip('-')[:80]
