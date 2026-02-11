"""
Domain models for the Zotero Document AI Pipeline.

This module defines the core data structures that represent the flow of
information through the OCR pipeline, from processing outcomes to extracted
content to Zotero note payloads.
These models provide type safety and clear documentation for the data transformations
that occur during document processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessingResult:
    """Represents the outcome of processing a single Zotero item.

    This model tracks the overall success status and detailed metrics for a single
    item processed by the pipeline, including PDF processing counts, page extraction
    statistics, and any errors encountered. Used for summary reporting and logging.
    """

    item_key: str
    """Zotero item key identifier. Uniquely identifies the item in the Zotero
    library."""

    item_title: str
    """Item title for logging and display purposes. Provides human-readable
    context for reports."""

    success: bool
    """Overall processing success status. True if all operations completed
    successfully, False otherwise."""

    pdfs_processed: int
    """Count of PDFs processed for this item. An item may have multiple
    attached PDFs."""

    pages_extracted: int
    """Total pages extracted across all PDFs for this item. Sum of all pages
    from all processed PDFs."""

    notes_created: int
    """Number of notes successfully created in Zotero. Each page typically
    results in one note."""

    errors: list[str]
    """List of error messages encountered during processing. Empty list if
    processing was successful."""

    processing_time: float
    """Processing duration in seconds. Includes time for PDF downloads, OCR
    processing, and note creation."""

    page_contents: dict[str, list[PageContent]] | None = None
    """Optional mapping of PDF filename to list of PageContent objects. Used
    for disk persistence of markdown content. None if not collected."""

    tree_structures: dict[str, DocumentTree] | None = None
    """Optional mapping of PDF filename to DocumentTree objects. Used for disk
    persistence of tree structures. None if not collected."""


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page.

    This model is the intermediate format between Mistral OCR output and Zotero
    note creation. It contains all the structured data extracted from a single
    page, including markdown text, images, and tables. This format allows for
    processing and formatting before note creation.
    """

    page_number: int
    """1-indexed page number within the PDF. Used for note titles and organization."""

    markdown: str
    """Extracted markdown content from OCR. Contains the main text and
    structure of the page. This markdown is later converted to HTML when
    creating Zotero notes for rich text rendering."""

    has_content: bool
    """Flag indicating if page contains extractable content. False for blank
    or image-only pages without OCR text."""

    images: list[str] = field(default_factory=list)
    """List of base64-encoded images from the page. Optional field, empty
    list if no images were extracted."""

    tables: list[str] = field(default_factory=list)
    """List of extracted tables in markdown format. Optional field, empty
    list if no tables were found on the page."""


@dataclass
class UploadedDocument:
    """Represents metadata for an uploaded document awaiting OCR processing.

    This model tracks uploaded documents in batch processing workflows, storing
    both the OCR provider's document ID and contextual metadata for tracking.
    Used primarily for batch upload/processing coordination.
    """

    doc_id: str
    """Document identifier from OCR provider (e.g., Mistral file_id). Used
    for subsequent processing operations."""

    filename: str
    """Original PDF filename. Used for logging and error reporting."""

    upload_time: float
    """Unix timestamp when document was uploaded. Used for tracking and
    timeout detection."""

    item_key: str | None = None
    """Zotero item key that owns this document. Used for associating results
    back to source items. Optional for batch uploads without context."""

    attachment_key: str | None = None
    """Zotero attachment key for this specific PDF. Used for tracking which
    attachment was processed. Optional for batch uploads without context."""


@dataclass
class NotePayload:
    """Base class for Zotero note payloads ready for creation via the API.

    This model structures all the data needed to create a single note in Zotero from
    processed page content. The content field stores markdown format internally, which
    is automatically converted to HTML during note creation for rich text display in
    Zotero's TinyMCE editor. Notes are created in batches (up to 50 notes per API call
    per Zotero limits) to optimize API usage and processing throughput.

    This class serves as the base for specialized note types (figures, tables) while
    maintaining backward compatibility. All specialized note types inherit these base
    fields and add type-specific identifiers.
    """

    parent_item_key: str
    """Zotero item key that will be the parent of this note. The note will
    be attached as a child item."""

    pdf_filename: str
    """Source PDF filename for identification. Used in note titles and for
    tracking the source document."""

    content: str
    """Formatted note content in markdown format. This markdown is
    automatically converted to HTML when creating Zotero notes via
    ZoteroClient.create_notes_batch() for rich text display in Zotero's
    TinyMCE editor. The conversion supports GFM features including tables,
    code blocks, emphasis, lists, and base64 image data URIs."""

    title: str
    """Note title following multi-line format: "Page {page_number} -
    \\n{pdf_filename} -\\nOCR". Provides clear identification in Zotero."""

    page_number: int | None = None
    """Page number within the specific PDF. None for all-at-once mode, int
    for page-by-page mode."""


@dataclass
class TreeNode:
    """Represents a single node in a document tree structure.

    This model provides a recursive tree structure for representing document
    hierarchies, such as table of contents, section headings, or nested
    document structures. Each node can contain child nodes, creating a tree
    that represents the document's organizational structure. Nodes can
    optionally include text content, summaries, and page references for
    navigation and content extraction.

    The recursive structure allows for arbitrary nesting depth, supporting
    complex document hierarchies. The to_dict() method enables JSON
    serialization of the entire tree structure for persistence and API
    communication.
    """

    node_id: str
    """Unique identifier for this node within the document tree. Used for referencing
    and navigation within the tree structure."""

    title: str
    """Node title or heading text. Represents the section heading, chapter title, or
    other hierarchical label for this node."""

    page_index: int
    """0-indexed page number where this node appears in the document. Used for
    navigation and content location within the PDF."""

    text: str | None = None
    """Optional full text content associated with this node. When included, contains
    the complete text content for this section or node. Excluded by default to reduce
    storage size and improve performance."""

    summary: str | None = None
    """Optional summary or description of this node's content. Provides a condensed
    overview of the node's content without including the full text."""

    nodes: list[TreeNode] = field(default_factory=list)
    """List of child nodes, creating the recursive tree structure. Empty list for leaf
    nodes. Child nodes can themselves contain nested children, allowing for arbitrary
    depth hierarchies."""

    def to_dict(self) -> dict[str, Any]:
        """Convert this node and all child nodes to dictionary format for JSON
        serialization.

        Recursively converts the node and all descendants to a dictionary structure that
        can be serialized to JSON. This method handles the recursive nature of the tree
        by calling to_dict() on each child node.

        Returns:
            Dictionary representation of this node and all descendants, suitable for
            JSON serialization. Includes all fields that are not None, with child nodes
            recursively converted to dictionaries.
        """
        result: dict[str, Any] = {
            "node_id": self.node_id,
            "title": self.title,
            "page_index": self.page_index,
        }

        if self.text is not None:
            result["text"] = self.text

        if self.summary is not None:
            result["summary"] = self.summary

        if self.nodes:
            result["nodes"] = [node.to_dict() for node in self.nodes]

        return result


@dataclass
class DocumentTree:
    """Represents a complete document tree structure with metadata.

    This model encapsulates a document's hierarchical structure as a tree of
    TreeNode objects, along with document metadata (ID, name, description).
    The tree structure represents the document's organization, such as table
    of contents, section hierarchy, or nested document structure. This format
    enables navigation, content extraction, and structured representation of
    document organization.

    The tree is typically extracted during OCR processing when tree structure extraction
    is enabled in the configuration. The structure is saved to disk as JSON for later
    retrieval and use in document navigation and content organization workflows.
    """

    doc_id: str
    """Document identifier, typically matching the OCR provider's document ID or
    the source PDF filename. Used for associating the tree structure with the
    source document."""

    doc_name: str
    """Human-readable document name or filename. Used for display and identification
    purposes in tree structure outputs and navigation interfaces."""

    description: str | None = None
    """Optional document description or metadata. Provides additional context about
    the document's content, purpose, or structure."""

    nodes: list[TreeNode] = field(default_factory=list)
    """List of root-level tree nodes. These nodes form the top level of the document
    hierarchy, with each node potentially containing nested child nodes. Empty list
    indicates no tree structure was extracted or the document has no hierarchical
    organization."""

    def to_dict(self) -> dict[str, Any]:
        """Convert this document tree to dictionary format for JSON serialization.

        Converts the document tree, including all metadata and recursively converting
        all nodes to dictionaries. This method enables persistence of tree structures
        to disk as JSON files, typically saved as {pdf_name}_tree_structure.json in
        the item's output directory.

        Returns:
            Dictionary representation of the document tree, suitable for JSON
            serialization. Includes doc_id, doc_name, optional description, and
            recursively converted node structures.
        """
        result: dict[str, Any] = {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
        }

        if self.description is not None:
            result["description"] = self.description

        if self.nodes:
            result["nodes"] = [node.to_dict() for node in self.nodes]

        return result
