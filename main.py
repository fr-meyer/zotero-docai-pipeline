"""Main entry point for Zotero Document AI Pipeline."""

import logging
import os
import sys

import hydra
from omegaconf import DictConfig

from zotero_docai_pipeline.cli.commands import dry_run_command, process_command
from zotero_docai_pipeline.clients.exceptions import (
    OCRClientError,
    ZoteroClientError,
)
from zotero_docai_pipeline.clients.mistral_client import MistralClient
from zotero_docai_pipeline.clients.ocr_client import OCRClient
from zotero_docai_pipeline.clients.pageindex_client import PageIndexClient
from zotero_docai_pipeline.clients.pageindex_tree_client import PageIndexTreeClient
from zotero_docai_pipeline.clients.zotero_client import ZoteroClient
from zotero_docai_pipeline.domain.config import (
    AppConfig,
    ConfigError,
    DownloadConfig,
    MistralOCRConfig,
    OCRProviderConfig,
    PageIndexOCRConfig,
    ProcessingConfig,
    RetryConfig,
    StorageConfig,
    TreeStructureConfig,
    ZoteroConfig,
    register_configs,
)
from zotero_docai_pipeline.domain.tree_processor import TreeStructureProcessor
from zotero_docai_pipeline.utils.logging import setup_logging


def initialize_clients(
    cfg: AppConfig, logger: logging.Logger
) -> tuple[ZoteroClient, OCRClient]:
    """Initialize Zotero and OCR clients from configuration.

    Args:
        cfg: Application configuration object
        logger: Logger instance

    Returns:
        Tuple of (zotero_client, ocr_client)
    """
    logger.info("Initializing Zotero client...")
    zotero_client = ZoteroClient(cfg.zotero)
    logger.info("Zotero client initialized successfully")

    logger.info("Initializing OCR client...")
    provider = cfg.ocr.provider
    ocr_client: OCRClient
    if provider == "mistral":
        if not isinstance(cfg.ocr, MistralOCRConfig):
            raise ValueError(
                "OCR config is not MistralOCRConfig for provider 'mistral'"
            )
        ocr_client = MistralClient(cfg.ocr)
    elif provider == "pageindex":
        if not isinstance(cfg.ocr, PageIndexOCRConfig):
            raise ValueError(
                "OCR config is not PageIndexOCRConfig for provider 'pageindex'"
            )
        ocr_client = PageIndexClient(cfg.ocr)
    else:
        raise ValueError(f"Unknown OCR provider: {provider}")

    logger.info("OCR client initialized successfully")

    return zotero_client, ocr_client


def validate_tree_config(cfg: AppConfig) -> None:
    """Validate tree structure configuration requirements.

    Ensures that when tree structure extraction is enabled, the configuration
    meets all requirements:
    - PageIndex API credentials must be available (either from OCR config if provider
      is PageIndex, or from a separate tree provider configuration)

    This validation is called after config conversion but before client initialization
    to catch configuration errors early. Validation is skipped entirely when
    processing.dry_run is True since dry-run mode doesn't initialize tree components.

    Note: Tree structure extraction can work with any OCR provider (e.g., Mistral)
    as long as PageIndex API credentials are available for tree processing.

    Args:
        cfg: Application configuration object

    Raises:
        ConfigError: If tree structure is enabled but requirements are not met.
            Error messages include actionable guidance for fixing the configuration.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Validating tree structure configuration")

    # Check if tree structure extraction is enabled
    if not cfg.tree_structure.enabled:
        return

    # Check if PageIndex API credentials are available for tree processing
    # Tree processing requires PageIndex API credentials regardless of OCR provider
    # For PageIndex OCR provider, credentials come from OCR config
    # For other providers (e.g., Mistral), tree credentials would need to be provided
    # separately (currently only available via PageIndexOCRConfig)
    has_pageindex_api_key = False

    if isinstance(cfg.ocr, PageIndexOCRConfig):
        has_pageindex_api_key = bool(cfg.ocr.api_key and cfg.ocr.api_key.strip())

    # Only validate/require PageIndex API key if tree is enabled
    # Don't fail for non-PageIndex OCR providers - tree processing will be skipped
    # during initialization if credentials aren't available
    if not has_pageindex_api_key:
        # Don't raise error - allow pipeline to proceed, tree processing will be skipped
        # This allows Mistral flows to work without tree processing
        logger.debug(
            "Tree structure extraction enabled but PageIndex API "
            "credentials not found. Tree processing will be skipped "
            "during initialization."
        )

    logger.debug("Tree structure configuration validated successfully")


def validate_flags(cfg: AppConfig) -> None:
    """Validate flag configuration compatibility.

    Ensures that configuration flags are compatible and at least one operation
    is enabled. This validation is called after AppConfig construction but before
    client initialization to catch configuration errors early.

    Validation rules:
    1. Mutually exclusive flags: processing.dry_run and download.enabled cannot
       both be True. Dry-run mode is for testing configuration without actual
       operations, while download is an actual operation.
    2. At least one operation enabled: Either download.enabled or ocr.enabled
       must be True. The pipeline requires at least one operation to perform.

    Args:
        cfg: Application configuration object

    Raises:
        ConfigError: If flag combinations are invalid. Error messages include
            actionable guidance for fixing the configuration.
    """
    logger = logging.getLogger(__name__)
    logger.debug("Validating flag configuration")

    # Check mutually exclusive flags: dry_run and download.enabled
    if cfg.processing.dry_run and cfg.download.enabled:
        raise ConfigError(
            "Invalid configuration: dry_run mode cannot be used with download feature. "
            "Set processing.dry_run=false or download.enabled=false."
        )

    # Check at least one operation must be enabled
    if not cfg.download.enabled and not cfg.ocr.enabled:
        raise ConfigError(
            "Invalid configuration: at least one operation must be enabled. "
            "Set download.enabled=true or ocr.enabled=true."
        )

    logger.debug("Flag configuration validated successfully")


def initialize_tree_processor(
    cfg: AppConfig,
    logger: logging.Logger,
) -> TreeStructureProcessor | None:
    """Initialize TreeStructureProcessor based on configuration.

    Initializes tree processor when tree structure extraction is enabled and
    PageIndex API credentials are available. Tree processing can work with any
    OCR provider (e.g., Mistral) as long as PageIndex credentials are provided
    for tree extraction.

    Credential lookup strategy:
    - Priority 1: If OCR provider is PageIndex, extract credentials
      from OCR config
    - Priority 2: If OCR provider is not PageIndex, read
      PAGEINDEX_API_KEY from environment
    - When credentials come from environment, use default base URL
      https://api.pageindex.ai

    Args:
        cfg: Application configuration object
        logger: Logger instance

    Returns:
        Optional TreeStructureProcessor instance. Returns None when tree
        structure extraction is disabled or tree credentials are missing.
    """
    # Check if tree structure extraction is enabled
    if not cfg.tree_structure.enabled:
        logger.debug("Tree structure processing disabled")
        return None

    # Check if tree provider is PageIndex
    if cfg.tree_structure.provider != "pageindex":
        logger.debug(
            f"Tree structure provider '{cfg.tree_structure.provider}' not supported"
        )
        return None

    # Two-tier credential lookup strategy
    api_key = None
    base_url = None
    credential_source = None

    # Priority 1: Extract credentials from OCR config if OCR provider is PageIndex
    if isinstance(cfg.ocr, PageIndexOCRConfig):
        api_key = cfg.ocr.api_key
        base_url = cfg.ocr.base_url
        credential_source = "OCR config"
        logger.debug("Using PageIndex credentials from OCR config")
    else:
        # Priority 2: Read credentials from environment when OCR
        # provider is not PageIndex
        api_key = os.getenv("PAGEINDEX_API_KEY")
        if api_key:
            base_url = "https://api.pageindex.ai"
            credential_source = "environment"
            logger.debug(
                "Using PageIndex credentials from PAGEINDEX_API_KEY "
                "environment variable"
            )

    # Only skip if tree credentials are missing
    if not api_key or not api_key.strip():
        if isinstance(cfg.ocr, MistralOCRConfig):
            logger.warning(
                "Tree structure extraction enabled with provider "
                "'pageindex' but PAGEINDEX_API_KEY environment variable "
                "not set. Set PAGEINDEX_API_KEY for tree processing "
                "with Mistral OCR."
            )
        else:
            logger.warning(
                "Tree structure extraction is enabled but PageIndex API "
                "key is not available. Tree extraction will be skipped. "
                "Provide PageIndex API credentials for tree processing."
            )
        return None

    # Use default base_url if not provided
    if not base_url or not base_url.strip():
        base_url = "https://api.pageindex.ai"
        logger.debug(f"Using default PageIndex base URL: {base_url}")

    logger.info(
        f"Initializing tree structure processor "
        f"(credentials from {credential_source})..."
    )

    tree_client = PageIndexTreeClient(
        cfg.tree_structure,
        base_url,
        api_key,
    )
    tree_processor = TreeStructureProcessor(tree_client)

    logger.info("Tree structure processor initialized successfully")
    return tree_processor


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    """Main entry point for the pipeline.

    Args:
        cfg: Hydra configuration object

    Returns:
        Exit code: 0 for success, 1 for partial failure, 2 for complete failure
    """
    # Register structured configs with Hydra
    register_configs()

    # Setup logging
    logger = setup_logging()

    try:
        # Check for old config structure migration (defensive check -
        # Hydra should catch this first)
        # This provides better error messages if old config structure
        # somehow gets through
        defaults = cfg.get("defaults", [])
        if defaults and any("mistral: default" in str(d) for d in defaults):
            raise ConfigError(
                "Migration needed: Old 'mistral: default' config structure "
                "detected. Please update your config defaults to use "
                "'ocr: default' or 'ocr: mistral' instead. "
                "See SPECS_BACKWARD_COMPATIBILITY.md for migration details."
            )

        # Convert DictConfig to structured AppConfig
        ocr_provider = cfg.ocr.provider
        if ocr_provider == "mistral":
            ocr_config = MistralOCRConfig(**cfg.ocr)
        elif ocr_provider == "pageindex":
            ocr_config = PageIndexOCRConfig(**cfg.ocr)
        else:
            ocr_config = OCRProviderConfig(**cfg.ocr)

        tree_structure_config = TreeStructureConfig(**cfg.tree_structure)

        # Construct RetryConfig from DictConfig before passing to DownloadConfig
        retry_config = RetryConfig(**cfg.download.retry)
        download_kw = {k: v for k, v in cfg.download.items() if k != "retry"}
        app_cfg = AppConfig(
            zotero=ZoteroConfig(**cfg.zotero),
            ocr=ocr_config,
            processing=ProcessingConfig(**cfg.processing),
            storage=StorageConfig(**cfg.storage),
            tree_structure=tree_structure_config,
            download=DownloadConfig(retry=retry_config, **download_kw),
        )

        # Validate flag configuration
        validate_flags(app_cfg)

        # Validate tree structure configuration (skip in dry-run mode)
        if not app_cfg.processing.dry_run:
            validate_tree_config(app_cfg)

        # Initialize clients
        zotero_client, ocr_client = initialize_clients(app_cfg, logger)

        # Initialize tree processor
        tree_processor = initialize_tree_processor(app_cfg, logger)

        # Log tree processor status
        if tree_processor is not None:
            logger.info("Tree structure processing enabled")
        elif app_cfg.tree_structure.enabled:
            logger.warning(
                "Tree structure processing requested but could not be initialized"
            )
        else:
            logger.debug("Tree structure processing disabled")

        # Route to appropriate command based on dry_run flag
        if app_cfg.processing.dry_run:
            exit_code = dry_run_command(app_cfg, logger, zotero_client)
        else:
            exit_code = process_command(
                app_cfg, logger, zotero_client, ocr_client, tree_processor
            )

        return exit_code

    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        return 3
    except (ZoteroClientError, OCRClientError) as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 3
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[call-arg]
