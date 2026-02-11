"""Command implementations for the Zotero Document AI Pipeline CLI.

This module contains the command functions that implement the dry-run and
processing workflows. These commands are called from the main entry point
after configuration validation and client initialization.
"""

import logging

from tabulate import tabulate

from zotero_docai_pipeline.clients.ocr_client import OCRClient
from zotero_docai_pipeline.clients.zotero_client import ZoteroClient
from zotero_docai_pipeline.domain.config import AppConfig
from zotero_docai_pipeline.domain.models import ProcessingResult
from zotero_docai_pipeline.domain.tree_processor import TreeStructureProcessor
from zotero_docai_pipeline.orchestration.pipeline import Pipeline
from zotero_docai_pipeline.orchestration.processor import ItemProcessor
from zotero_docai_pipeline.utils.logging import (
    _format_with_emoji,
    _supports_unicode,
    log_config_summary,
    log_error_summary,
    log_summary_table,
    log_timing_summary,
)


def dry_run_command(
    cfg: AppConfig, logger: logging.Logger, zotero_client: ZoteroClient
) -> int:
    """Execute dry-run mode to preview items without processing.

    Fetches items matching the configured tags and displays a preview table
    showing item titles and PDF counts without performing any actual processing.

    Args:
        cfg: Application configuration object
        logger: Logger instance for logging messages
        zotero_client: Initialized Zotero client instance

    Returns:
        Exit code: 0 for success
    """
    logger.info("Dry-run mode enabled - previewing items without processing")
    items = zotero_client.get_items_by_tag(
        cfg.zotero.tags.input, cfg.zotero.tags.output
    )

    # Count total PDFs across all items
    total_pdfs = sum(
        len(
            [
                att
                for att in item.get("attachments", [])
                if att.get("filename", "").lower().endswith(".pdf")
            ]
        )
        for item in items
    )

    log_config_summary(logger, len(items), cfg.ocr)
    logger.info(f"Total PDF attachments: {total_pdfs}")

    # Display preview table if items exist
    if items:
        logger.info("Preview of items to be processed:")
        # Create table header
        if _supports_unicode():
            logger.info(
                "╔════════════════════════════════════════════════════════════════╗"
            )
            logger.info(
                "║                    Dry-Run Preview                            ║"
            )
            logger.info(
                "╠════════════════════════════════════════════════════════════════╣"
            )
        else:
            logger.info("=" * 64)
            logger.info("                    Dry-Run Preview")
            logger.info("=" * 64)

        # Display items
        for item in items:
            pdf_count = len(
                [
                    att
                    for att in item.get("attachments", [])
                    if att.get("filename", "").lower().endswith(".pdf")
                ]
            )
            title = item.get("title", "Untitled")
            if _supports_unicode():
                logger.info(f"║ {title[:50]:<50} │ PDFs: {pdf_count:>3} ║")
            else:
                logger.info(f"  {title[:50]:<50} | PDFs: {pdf_count:>3}")

        # Display summary
        if _supports_unicode():
            logger.info(
                "╠════════════════════════════════════════════════════════════════╣"
            )
            logger.info(f"║ Total Items: {len(items):<47} ║")
            logger.info(f"║ Total PDFs:  {total_pdfs:<47} ║")
            logger.info(
                "╚════════════════════════════════════════════════════════════════╝"
            )
        else:
            logger.info("-" * 64)
            logger.info(f"Total Items: {len(items)}")
            logger.info(f"Total PDFs:  {total_pdfs}")
            logger.info("=" * 64)
    else:
        logger.info("No items found to process")

    return 0


def _determine_exit_code(
    summary: dict[str, int | float | list[ProcessingResult]],
) -> int:
    """Determine the appropriate exit code based on processing summary.

    Args:
        summary: Dictionary containing processing summary with keys:
            - failed_items: Number of items that failed processing
            - successful_items: Number of items that succeeded
            - total_items: Total number of items processed

    Returns:
        Exit code: 0 for success, 1 for partial failure, 2 for complete failure
    """
    failed_items = summary.get("failed_items", 0)
    successful_items = summary.get("successful_items", 0)
    total_items = summary.get("total_items", 0)

    # Type narrow to int
    if not isinstance(failed_items, int):
        failed_items = 0
    if not isinstance(successful_items, int):
        successful_items = 0
    if not isinstance(total_items, int):
        total_items = 0

    if failed_items == 0:
        return 0  # Success
    elif successful_items > 0:
        return 1  # Partial failure
    elif total_items > 0:
        return 2  # Complete failure
    else:
        return 0  # No items is not an error


def _display_download_summary(logger: logging.Logger, summary: dict) -> None:
    """Display summary for download-only mode.

    Displays a formatted table with download statistics including downloaded,
    skipped, and failed counts, along with timing information and skipped items.

    Args:
        logger: Logger instance for logging messages
        summary: Dictionary containing download processing summary with keys:
            - total_pdfs_downloaded: Number of PDFs successfully downloaded
            - total_pdfs_skipped: Number of PDFs skipped
            - total_pdfs_failed: Number of PDFs that failed to download
            - total_time: Total execution time in seconds
            - skipped_items: Optional number of items skipped during discovery
    """
    # Summary header
    logger.info("")
    formatted_header = _format_with_emoji("Download Summary:", "📥", "[DOWNLOAD]")
    logger.info(formatted_header)

    # Extract metrics from summary dictionary
    downloaded = summary.get("total_pdfs_downloaded", 0)
    skipped = summary.get("total_pdfs_skipped", 0)
    failed = summary.get("total_pdfs_failed", 0)
    total = downloaded + skipped + failed

    # Build table data
    table_data = [
        ["Downloaded", downloaded],
        ["Skipped", skipped],
        ["Failed", failed],
        ["Total", total],
    ]

    # Format and log table
    tablefmt = "grid" if _supports_unicode() else "simple"
    table_str = tabulate(table_data, headers=["Status", "Count"], tablefmt=tablefmt)
    logger.info(table_str)

    # Timing summary
    log_timing_summary(logger, summary.get("total_time", 0))

    # Skipped items
    skipped_items = summary.get("skipped_items", 0)
    if skipped_items > 0:
        message = f"Skipped: {skipped_items} items (already processed)"
        formatted_message = _format_with_emoji(message, "⏭️", "[SKIP]")
        logger.info("")
        logger.info(formatted_message)


def _display_combined_summary(logger: logging.Logger, summary: dict) -> None:
    """Display summary for download+OCR combined mode.

    Displays both download statistics and OCR results in a unified view,
    including per-item OCR results, timing information, and error details.

    Args:
        logger: Logger instance for logging messages
        summary: Dictionary containing combined download and OCR processing summary
            with keys:
            - total_pdfs_downloaded: Number of PDFs successfully downloaded
            - total_pdfs_skipped: Number of PDFs skipped
            - total_pdfs_failed: Number of PDFs that failed to download
            - results: List of ProcessingResult objects from OCR processing
            - skipped_items: Optional number of items skipped during discovery
            - total_time: Total execution time in seconds
    """
    # Download section
    logger.info("")
    formatted_header = _format_with_emoji("Download Summary:", "📥", "[DOWNLOAD]")
    logger.info(formatted_header)

    # Extract download metrics
    downloaded = summary.get("total_pdfs_downloaded", 0)
    skipped = summary.get("total_pdfs_skipped", 0)
    failed = summary.get("total_pdfs_failed", 0)

    # Build download table data
    download_table_data = [
        ["Downloaded", downloaded],
        ["Skipped", skipped],
        ["Failed", failed],
    ]

    # Format and log download table
    tablefmt = "grid" if _supports_unicode() else "simple"
    download_table_str = tabulate(
        download_table_data, headers=["Status", "Count"], tablefmt=tablefmt
    )
    logger.info(download_table_str)

    # OCR section
    logger.info("")
    ocr_header = _format_with_emoji("OCR Summary:", "🔍", "[OCR]")
    logger.info(ocr_header)
    log_summary_table(
        logger, summary.get("results", []), summary.get("skipped_items", 0)
    )

    # Timing and errors
    log_timing_summary(logger, summary.get("total_time", 0))
    log_error_summary(logger, summary.get("results", []))


def process_command(
    cfg: AppConfig,
    logger: logging.Logger,
    zotero_client: ZoteroClient,
    ocr_client: OCRClient,
    tree_processor: TreeStructureProcessor | None = None,
) -> int:
    """Execute the full pipeline processing workflow.

    Initializes the processor and pipeline, executes the processing workflow,
    and displays comprehensive summary information including per-item results,
    timing, and error details. Supports download-only, download+OCR, and OCR-only
    modes with appropriate summary display for each mode.

    Args:
        cfg: Application configuration object
        logger: Logger instance for logging messages
        zotero_client: Initialized Zotero client instance
        ocr_client: Initialized OCR client instance
        tree_processor: Optional tree structure processor instance

    Returns:
        Exit code: 0 for success, 1 for partial failure, 2 for complete failure
    """
    logger.info("Starting pipeline execution...")
    processor = ItemProcessor(zotero_client, ocr_client, cfg.processing)
    pipeline = Pipeline(
        zotero_client,
        ocr_client,
        processor,
        cfg.zotero,
        cfg.processing,
        cfg.storage,
        cfg.tree_structure,
        cfg.ocr,
        cfg.download,
        tree_processor=tree_processor,
    )
    summary = pipeline.run()

    # Determine which mode is enabled for appropriate summary display
    download_only = cfg.download.enabled and not cfg.ocr.enabled
    download_and_ocr = cfg.download.enabled and cfg.ocr.enabled

    # Route to appropriate summary display based on mode
    if download_only:
        # Display download-only summary with download statistics
        _display_download_summary(logger, summary)
    elif download_and_ocr:
        # Display combined summary with both download and OCR results
        _display_combined_summary(logger, summary)
    else:
        # Display OCR-only summary (backward compatible)
        results = summary.get("results", [])
        skipped_items = summary.get("skipped_items", 0)
        total_time = summary.get("total_time", 0.0)

        # Type narrow to expected types
        if not isinstance(results, list):
            results = []
        if not isinstance(skipped_items, int):
            skipped_items = 0
        if not isinstance(total_time, (int, float)):
            total_time = 0.0

        log_summary_table(logger, results, skipped_items)
        log_timing_summary(logger, float(total_time))
        log_error_summary(logger, results)

    # Determine exit code based on summary
    return _determine_exit_code(summary)
