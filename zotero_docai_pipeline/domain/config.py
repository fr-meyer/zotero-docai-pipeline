"""
Configuration dataclasses for the Zotero Document AI Pipeline.

This module defines type-safe configuration schemas using Python dataclasses.
These schemas are registered with Hydra to enable validation and IDE autocomplete
support for configuration values.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


class ConfigError(Exception):
    """Configuration error for the Zotero Document AI Pipeline.

    Raised when configuration values are invalid or inconsistent. Using a
    dedicated exception type makes it easier to distinguish configuration
    problems from other runtime errors.
    """


@dataclass
class ZoteroTagsConfig:
    """Configuration for Zotero tag-based workflow.

    Tags are used to mark items for processing and track processing status.
    """

    input: str = "docai"
    """Tag that marks items for processing. Items with this tag will be
    processed by the pipeline."""

    output: str = "docai-processed"
    """Tag added to successfully processed items. Prevents reprocessing of
    already processed items."""

    error: str = "docai-error"
    """Tag added to items that fail processing. Use this tag to easily
    identify and review failed items."""


@dataclass
class ZoteroConfig:
    """Configuration for Zotero API integration.

    Contains settings for connecting to and interacting with the Zotero API,
    including library identification, authentication, and tag-based workflow.
    """

    library_id: str
    """Zotero library ID. Find this in your Zotero web library URL:
    https://www.zotero.org/users/{library_id}"""

    api_key: str
    """Zotero API key. Obtain from https://www.zotero.org/settings/keys"""

    tags: ZoteroTagsConfig = field(default_factory=ZoteroTagsConfig)
    """Tag-based workflow configuration. Defines tags for input, output, and
    error states."""

    error_tagging_enabled: bool = True
    """Whether to add error tags to failed items. Makes it easy to identify
    failed items in Zotero."""

    def __post_init__(self) -> None:
        """Validate that library_id is not empty."""
        if not self.library_id or not self.library_id.strip():
            raise ConfigError(
                "library_id is required and cannot be empty. "
                "Find your library ID in your Zotero web library URL: "
                "https://www.zotero.org/users/{library_id}"
            )


@dataclass
class OCRProviderConfig:
    """Base configuration class for OCR provider implementations.

    All OCR provider configs must inherit from this base class and implement
    provider-specific fields. The provider field is used for runtime selection.
    """

    enabled: bool = True
    """Whether OCR processing is enabled. When False, only PDF download will occur."""

    provider: str = "mistral"
    """OCR provider identifier. Options: 'mistral', 'pageindex'"""


@dataclass
class MistralOCRConfig(OCRProviderConfig):
    """Configuration for Mistral AI OCR API.

    Contains settings for the Mistral AI OCR service, including model selection,
    output format preferences, and extraction options.
    """

    provider: str = "mistral"
    """OCR provider identifier. Set to 'mistral' for Mistral OCR."""

    api_key: str = ""
    """Mistral AI API key. Obtain from https://console.mistral.ai"""

    model: str = "mistral-ocr-latest"
    """OCR model to use. Default uses the latest available model."""

    table_format: str = "markdown"
    """Format for extracted tables. Markdown is recommended for readability
    in Zotero notes."""

    include_image_base64: bool = True
    """Whether to include images as base64-encoded strings in notes."""

    embed_images_in_markdown: bool = False
    """Whether to embed images inline in markdown as base64 data URIs.
    When false (default), image placeholders (e.g., ![img-0.jpeg](img-0.jpeg))
    remain in markdown and images are available separately in
    PageContent.images list. Requires include_image_base64=True to have any
    effect."""

    extract_header: bool = False
    """Whether to extract document headers. May increase processing time."""

    extract_footer: bool = False
    """Whether to extract document footers. May increase processing time."""


@dataclass
class PageIndexOCRConfig(OCRProviderConfig):
    """Configuration for PageIndex OCR API.

    Contains settings for the PageIndex OCR service, including API authentication,
    batch processing options, and polling configuration with exponential backoff.
    """

    provider: str = "pageindex"
    """OCR provider identifier. Set to 'pageindex' for PageIndex OCR."""

    api_key: str = ""
    """PageIndex API key. Obtain from PageIndex console."""

    base_url: str = "https://api.pageindex.ai"
    """Base URL for PageIndex API. Default points to production endpoint."""

    use_sdk: bool = False
    """Whether to use PageIndex Python SDK instead of direct HTTP API requests.

    When True: Uses official PageIndex SDK (pageindex package) for all
    PageIndex operations. The SDK provides automatic API versioning, simplified
    error handling, and access to SDK-specific features.

    When False (default): Uses direct HTTP requests via the requests library.
    This is the default mode for backward compatibility with existing
    deployments.

    Both modes provide identical functionality and performance. The choice
    between SDK and HTTP API mode is a matter of preference and operational
    requirements.

    Requirements:
    - SDK mode requires the pageindex package to be installed
      (included in environment.yml)
    - Both modes require a valid PageIndex API key

    Default: false (HTTP API mode) for backward compatibility.
    """

    batch_upload_enabled: bool = True
    """Whether to enable batch upload optimization. When enabled, documents
    are uploaded in batch mode and then polled sequentially, improving
    efficiency for multiple documents."""

    polling_initial_interval: int = 2
    """Starting interval in seconds for exponential backoff polling. The
    interval doubles on each retry until reaching polling_max_interval."""

    polling_max_interval: int = 30
    """Maximum interval cap in seconds for exponential backoff polling.
    Prevents intervals from growing too large and ensures reasonable response
    times."""

    polling_timeout: int = 300
    """Overall timeout in seconds for polling operations. If polling exceeds
    this duration, the operation will be aborted. Default 300 seconds
    (5 minutes) provides sufficient time for typical document processing."""

    polling_max_attempts: int = 20
    """Maximum number of polling attempts per document. With exponential
    backoff starting at polling_initial_interval, this provides adequate
    coverage for document processing while preventing infinite retry loops."""

    polling_batch_delay: int = 1
    """Delay in seconds between processing documents in a batch. Prevents API
    rate limiting when polling multiple documents sequentially."""


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""

    max_attempts: int = 3
    """Maximum number of retry attempts before giving up."""

    initial_delay: int = 2
    """Initial delay in seconds before first retry."""

    backoff_multiplier: float = 2.0
    """Multiplier for exponential backoff. Delay doubles by default on each retry."""

    max_delay: int = 16
    """Maximum delay cap in seconds to prevent excessively long waits."""

    def __post_init__(self) -> None:
        """Validate retry configuration parameters."""
        if self.max_attempts <= 0:
            raise ConfigError("max_attempts must be greater than 0")
        if self.initial_delay <= 0:
            raise ConfigError("initial_delay must be greater than 0")
        if self.backoff_multiplier <= 0:
            raise ConfigError("backoff_multiplier must be greater than 0")
        if self.max_delay <= 0:
            raise ConfigError("max_delay must be greater than 0")
        if self.max_delay < self.initial_delay:
            raise ConfigError(
                "max_delay must be greater than or equal to initial_delay"
            )


@dataclass
class DownloadConfig:
    """Configuration for PDF download feature. Controls whether and how PDFs
    are downloaded from Zotero items."""

    enabled: bool = False
    """Whether PDF download feature is enabled. Default False for backward
    compatibility."""

    tag: str = "docai"
    """Zotero tag to filter items for download. Only items with this tag
    will be processed."""

    upload_folder: str = "./downloads"
    """Local directory path where downloaded PDFs will be saved."""

    preserve_filenames: bool = True
    """Whether to preserve original PDF filenames. When False, files are
    renamed based on Zotero item metadata."""

    create_subfolders: bool = False
    """Whether to create subfolders for organizing downloads (e.g., by
    collection or date)."""

    skip_existing: bool = True
    """Whether to skip downloading PDFs that already exist locally. Prevents
    redundant downloads."""

    max_concurrent_downloads: int = 5
    """Maximum number of concurrent download operations. Must be greater than 0."""

    retry: RetryConfig = field(default_factory=RetryConfig)
    """Retry configuration for failed download attempts."""

    def __post_init__(self) -> None:
        """Validate download configuration parameters."""
        if not self.tag or not self.tag.strip():
            raise ConfigError(
                "tag is required and cannot be empty or whitespace-only. "
                "Specify a Zotero tag to filter items for download."
            )
        if not self.upload_folder or not self.upload_folder.strip():
            raise ConfigError(
                "upload_folder is required and cannot be empty or whitespace-only. "
                "Specify a valid directory path for downloads."
            )
        if self.max_concurrent_downloads <= 0:
            raise ConfigError(
                "max_concurrent_downloads must be greater than 0. "
                "Specify a positive integer for concurrent downloads."
            )
        if not isinstance(self.retry, RetryConfig):
            raise ConfigError("retry must be an instance of RetryConfig")


@dataclass
class TreeStructureConfig:
    """Configuration for document tree structure extraction and processing.

    Controls whether and how document tree structures (such as table of contents,
    section hierarchies, or nested document structures) are extracted during OCR
    processing. Tree structures provide hierarchical organization of document content,
    enabling navigation and structured content extraction.

    When enabled, tree structures are extracted during OCR processing and saved to
    disk as JSON files in the format {item_dir}/{pdf_name}_tree_structure.json.
    The tree structure represents the document's organizational hierarchy, with nodes
    containing titles, page references, optional summaries, and optional text content.

    Tree structure extraction occurs automatically during OCR processing when enabled.
    The extraction is provider-specific (currently only PageIndex supports tree
    structure extraction). Performance may be impacted when including full text content
    in tree nodes, as this significantly increases the size of the tree structure.
    """

    enabled: bool = False
    """Whether tree structure extraction is enabled. When True, tree structures are
    extracted during OCR processing and saved to disk. When False (default), tree
    structure extraction is skipped entirely."""

    provider: str = "pageindex"
    """Tree structure provider identifier. Currently only 'pageindex' is supported
    for tree structure extraction. Other OCR providers may support tree extraction
    in future versions."""

    node_id: bool = True
    """Whether to include node IDs in tree structure. Node IDs provide unique
    identifiers for each node, enabling programmatic navigation and referencing
    within the tree structure."""

    summary: bool = True
    """Whether to include node summaries in tree structure. Summaries provide
    condensed overviews of node content without including full text, balancing
    information content with storage efficiency."""

    text: bool = False
    """Whether to include full text content in tree nodes. When True, each node
    includes the complete text content for its section, significantly increasing
    tree structure size. When False (default), only metadata (title, page index,
    summary) is included, reducing storage requirements and improving performance.
    Set to True only when full text content is required for downstream processing."""

    description: bool = True
    """Whether to include document description in tree structure. Document descriptions
    provide additional metadata about the document's content, purpose, or structure,
    enhancing the tree structure's informational value."""


@dataclass
class ProcessingConfig:
    """Configuration for document processing behavior.

    Controls how the pipeline processes documents, including batch sizes,
    error handling, and optional features.
    """

    dry_run: bool = False
    """Preview mode without actual processing. Useful for testing
    configuration."""

    batch_size: int = 50
    """Number of notes to process in each batch. Must not exceed 50 (Zotero
    API limit)."""

    skip_empty_pages: bool = True
    """Whether to skip pages with no content. Reduces clutter in Zotero library."""

    save_to_disk: bool = False
    """Whether to save processed content to disk. Useful for debugging."""

    note_size_threshold: int = 180000
    """Maximum note content size in bytes (character count) before triggering
    size validation. Default 180000 bytes provides safety margin below
    Zotero's 250KB limit to account for HTML encoding overhead. Notes
    exceeding this threshold will either be split automatically (if
    auto_split_oversized_notes is true) or raise NoteSizeExceededError.
    Recommendation: Keep default unless experiencing size-related API
    failures."""

    auto_split_oversized_notes: bool = True
    """Whether to automatically split oversized notes into smaller chunks.
    When enabled, page notes exceeding note_size_threshold are split into
    multiple sequential notes (e.g., 'Page 1 - Part 1', 'Page 1 - Part 2').
    Figure and table notes cannot be split and will raise
    NoteSizeExceededError if oversized. Set to false to fail fast on
    oversized content for manual review."""

    extraction_mode: str = "all_at_once"
    """How to organize extracted content into notes.
Options:
- 'all_at_once': Create a single note containing all pages (default)
- 'page_by_page': Create one note per page (legacy behavior, opt-in)

When 'all_at_once' is selected (default):
- All pages are concatenated into one note
- Page separators (e.g., '--- Page N ---') are inserted between pages
- Size limits still apply (auto-split if content exceeds threshold)
- Empty pages are still filtered if skip_empty_pages=True

When 'page_by_page' is selected (legacy):
- One note is created per page
- Each note has its own title with page number
- Useful for users who prefer page-by-page organization
"""

    cleanup_uploaded_files: bool = False
    """Controls whether uploaded OCR files are deleted after processing.

    When True: Files are cleaned up from OCR provider storage after
    processing completes. This helps avoid storage costs in production
    environments. When False: Files remain in OCR provider storage.

    When False (default): Files are retained on OCR provider storage for
    debugging/review. This is useful during development and troubleshooting to
    inspect uploaded files.

    Recommendation: Set to True in production to avoid storage costs, False
    for debugging.
    """


@dataclass
class StorageConfig:
    """Configuration for optional disk storage features.

    These features are primarily useful for debugging and development.
    All storage is optional and can be disabled for production use.
    """

    base_dir: str = "./data/ocr_output"
    """Root directory for optional disk storage. Created automatically if
    it doesn't exist."""

    save_pdfs: bool = False
    """Whether to save downloaded PDFs to disk. Can use significant disk space."""

    save_manifests: bool = True
    """Whether to save processing metadata as JSON. Useful for debugging and
    tracking."""


@dataclass
class AppConfig:
    """Top-level application configuration.

    Combines all configuration groups into a single type-safe configuration
    object. This is the configuration class that Hydra will instantiate and
    pass to the main function.
    """

    zotero: ZoteroConfig
    """Zotero API configuration."""

    ocr: OCRProviderConfig
    """OCR provider configuration (Mistral, PageIndex, etc.)."""

    processing: ProcessingConfig
    """Processing behavior configuration."""

    storage: StorageConfig
    """Storage options configuration."""

    tree_structure: TreeStructureConfig = field(default_factory=TreeStructureConfig)
    """Tree structure extraction and processing configuration."""

    download: DownloadConfig = field(default_factory=DownloadConfig)
    """PDF download feature configuration."""


def register_configs() -> None:
    """Register structured configs with Hydra.

    This function must be called before Hydra initializes to enable type-safe
    configuration validation and IDE autocomplete support.
    """
    cs = ConfigStore.instance()

    # Register config groups with names matching YAML defaults
    cs.store(group="zotero", name="default", node=ZoteroConfig)

    # Register OCR provider configs
    cs.store(group="ocr", name="mistral", node=MistralOCRConfig)
    cs.store(group="ocr", name="pageindex", node=PageIndexOCRConfig)
    cs.store(group="ocr", name="default", node=OCRProviderConfig)

    cs.store(group="processing", name="default", node=ProcessingConfig)
    cs.store(group="storage", name="default", node=StorageConfig)
    cs.store(group="tree_structure", name="default", node=TreeStructureConfig)
    cs.store(group="download", name="default", node=DownloadConfig)

    # Register top-level config
    cs.store(name="config", node=AppConfig)
