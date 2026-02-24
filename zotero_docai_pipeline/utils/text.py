"""Text normalization utilities for the Zotero DocAI Pipeline."""


def normalize_title(title: str) -> str:
    """Normalize a title for case-insensitive, whitespace-insensitive comparison.

    Strips leading/trailing whitespace, collapses internal runs of whitespace
    to a single space, and lowercases the result.
    """
    return " ".join(title.strip().split()).lower()
