"""
Zotero Document AI Pipeline

Automate PDF-to-Markdown extraction for Zotero attachments using Mistral Document AI,
and save results as Zotero notes for Notero/Notion sync.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zotero-docai-pipeline")
except PackageNotFoundError:
    __version__ = "unknown"
