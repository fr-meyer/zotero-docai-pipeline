"""Logging utilities for the Zotero Document AI Pipeline.

This module provides structured logging functions that integrate with Hydra's
logging system and support unicode/emoji for user-friendly terminal output.
"""

import logging
import os
import sys

from tabulate import tabulate

from zotero_docai_pipeline.domain.config import OCRProviderConfig, PageIndexOCRConfig
from zotero_docai_pipeline.domain.models import ProcessingResult


def _supports_unicode() -> bool:
    """Detect if terminal supports unicode/emoji.

    Checks system encoding and environment variables to determine if the
    terminal can display unicode characters and emoji.

    Returns:
        True if terminal supports unicode, False otherwise
    """
    # Check for explicit ASCII-only mode
    if os.environ.get("FORCE_ASCII") == "1":
        return False

    # Check stdout encoding
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding is None:
        return False

    # Common unicode-capable encodings
    unicode_encodings = {"utf-8", "utf-16", "utf-32", "utf-8-sig"}
    return encoding.lower() in unicode_encodings


def _format_with_emoji(message: str, emoji: str, fallback: str) -> str:
    """Format message with emoji or fallback text.

    Args:
        message: The message text to format
        emoji: Unicode emoji character to prepend
        fallback: ASCII fallback text to use if unicode not supported

    Returns:
        Formatted message with emoji or fallback
    """
    if _supports_unicode():
        return f"{emoji} {message}"
    else:
        return f"{fallback} {message}"


def setup_logging() -> logging.Logger:
    """Initialize logging configuration for the pipeline.

    Returns the configured logger instance. Hydra automatically configures
    logging when @hydra.main() decorator is used, so this function primarily
    returns the logger instance for use throughout the pipeline.

    Returns:
        Configured logger instance ready for use

    Example:
        >>> logger = setup_logging()
        >>> logger.info("Pipeline started")
    """
    logger = logging.getLogger(__name__)

    # Hydra automatically configures logging, so we primarily work with
    # the existing logger configuration. We can add custom formatters if needed.
    # For now, return the logger as-is since Hydra handles the setup.

    return logger


def log_item_start(
    logger: logging.Logger, item_title: str, item_number: int, total_items: int
) -> None:
    """Log the start of processing an item.

    Formats a user-friendly message indicating which item is being processed
    with progress information.

    Args:
        logger: Logger instance to use for logging
        item_title: Title or identifier of the item being processed
        item_number: Current item number (1-indexed)
        total_items: Total number of items to process

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_item_start(logger, "My Document", 1, 10)
        # Output: "📄 Processing [1/10]: \"My Document\""
    """
    if _supports_unicode():
        message = f'📄 Processing [{item_number}/{total_items}]: "{item_title}"'
    else:
        message = f'[*] Processing [{item_number}/{total_items}]: "{item_title}"'

    logger.info(message)


def log_startup(logger: logging.Logger, message: str) -> None:
    """Log pipeline startup message.

    Formats a user-friendly startup message with emoji support.

    Args:
        logger: Logger instance to use for logging
        message: Startup message text

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_startup(logger, "Starting Zotero DocAI Pipeline")
        # Output: "🚀 Starting Zotero DocAI Pipeline" or "[START] Starting
        # Zotero DocAI Pipeline"
    """
    formatted_message = _format_with_emoji(message, "🚀", "[START]")
    logger.info(formatted_message)


def log_config_summary(
    logger: logging.Logger,
    item_count: int,
    ocr_config: OCRProviderConfig | None = None,
) -> None:
    """Log configuration summary with item count.

    Formats a user-friendly configuration summary message. When using PageIndex
    OCR provider, appends mode information (SDK mode or HTTP API) based on
    the use_sdk configuration.

    Args:
        logger: Logger instance to use for logging
        item_count: Number of items discovered for processing
        ocr_config: Optional OCR provider configuration. If provided and is a
            PageIndexOCRConfig instance, mode information (SDK mode or HTTP
            API) will be appended to the message.

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_config_summary(logger, 5)
        # Output: "📋 Discovered 5 items to process" or "[CONFIG] Discovered
        # 5 items to process"
        >>> log_config_summary(logger, 5, PageIndexOCRConfig(use_sdk=True))
        # Output: "📋 Discovered 5 items to process (SDK mode)" or "[CONFIG]
        # Discovered 5 items to process (SDK mode)"
        >>> log_config_summary(logger, 5, PageIndexOCRConfig(use_sdk=False))
        # Output: "📋 Discovered 5 items to process (HTTP API)" or "[CONFIG]
        # Discovered 5 items to process (HTTP API)"
    """
    message = f"Discovered {item_count} items to process"

    # Append mode suffix for PageIndex provider
    if ocr_config is not None and isinstance(ocr_config, PageIndexOCRConfig):
        if ocr_config.use_sdk:
            message += " (SDK mode)"
        else:
            message += " (HTTP API)"

    formatted_message = _format_with_emoji(message, "📋", "[CONFIG]")
    logger.info(formatted_message)


def log_skipped_item(logger: logging.Logger, item_title: str, reason: str) -> None:
    """Log a skipped item with reason.

    Formats a user-friendly message for items that are skipped during processing.

    Args:
        logger: Logger instance to use for logging
        item_title: Title or identifier of the skipped item
        reason: Reason why the item was skipped

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_skipped_item(logger, "My Document", "already processed")
        # Output: "⏭️ Skipped \"My Document\": already processed" or "[SKIP]
        # Skipped \"My Document\": already processed"
    """
    message = f'Skipped "{item_title}": {reason}'
    formatted_message = _format_with_emoji(message, "⏭️", "[SKIP]")
    logger.info(formatted_message)


def log_pdf_download_start(
    logger: logging.Logger, filename: str, pdf_number: int, total_pdfs: int
) -> None:
    """Log the start of PDF download.

    Formats a user-friendly message indicating PDF download initiation with progress.

    Args:
        logger: Logger instance to use for logging
        filename: Name of the PDF file being downloaded
        pdf_number: Current PDF number (1-indexed)
        total_pdfs: Total number of PDFs to download

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_pdf_download_start(logger, "document.pdf", 1, 2)
        # Output: "⬇️ Downloading PDF [1/2]: document.pdf..." or "[DOWNLOAD]
        # Downloading PDF [1/2]: document.pdf..."
    """
    message = f"Downloading PDF [{pdf_number}/{total_pdfs}]: {filename}..."
    formatted_message = _format_with_emoji(message, "⬇️", "[DOWNLOAD]")
    logger.info(formatted_message)


def log_pdf_download_success(
    logger: logging.Logger, filename: str, size_mb: float
) -> None:
    """Log successful PDF download.

    Formats a user-friendly message indicating successful PDF download with file size.

    Args:
        logger: Logger instance to use for logging
        filename: Name of the downloaded PDF file
        size_mb: File size in megabytes

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_pdf_download_success(logger, "document.pdf", 2.3)
        # Output: "✓ Downloaded (2.3 MB)" or "[OK] Downloaded (2.3 MB)"
    """
    message = f"Downloaded {filename} ({size_mb:.1f} MB)"
    formatted_message = _format_with_emoji(message, "✓", "[OK]")
    logger.info(formatted_message)


def log_ocr_start(logger: logging.Logger, filename: str) -> None:
    """Log the start of OCR processing.

    Formats a user-friendly message indicating OCR processing initiation.

    Args:
        logger: Logger instance to use for logging
        filename: Name of the PDF file being processed

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_ocr_start(logger, "document.pdf")
        # Output: "🔍 Starting OCR with Mistral..." or "[OCR] Starting OCR
        # with Mistral..."
    """
    message = f"Starting OCR with Mistral for {filename}..."
    formatted_message = _format_with_emoji(message, "🔍", "[OCR]")
    logger.info(formatted_message)


def log_ocr_success(
    logger: logging.Logger,
    pages_extracted: int,
    pages_with_content: int,
    pages_skipped: int,
) -> None:
    """Log successful OCR completion.

    Formats a user-friendly message indicating OCR completion with page statistics.

    Args:
        logger: Logger instance to use for logging
        pages_extracted: Total number of pages extracted
        pages_with_content: Number of pages with content
        pages_skipped: Number of empty pages skipped

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_ocr_success(logger, 75, 70, 5)
        # Output: "✓ OCR complete: 75 pages extracted (70 with content, 5
        # empty skipped)" or "[OK] OCR complete: 75 pages extracted (70 with
        # content, 5 empty skipped)"
    """
    message = (
        f"OCR complete: {pages_extracted} pages extracted "
        f"({pages_with_content} with content, {pages_skipped} empty skipped)"
    )
    formatted_message = _format_with_emoji(message, "✓", "[OK]")
    logger.info(formatted_message)


def log_note_creation_start(logger: logging.Logger, note_count: int) -> None:
    """Log the start of note creation.

    Formats a user-friendly message indicating note creation initiation.

    Args:
        logger: Logger instance to use for logging
        note_count: Number of notes to be created

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_note_creation_start(logger, 10)
        # Output: "📝 Creating 10 notes..." or "[NOTES] Creating 10 notes..."
    """
    message = f"Creating {note_count} notes..."
    formatted_message = _format_with_emoji(message, "📝", "[NOTES]")
    logger.info(formatted_message)


def log_note_creation_success(logger: logging.Logger, notes_created: int) -> None:
    """Log successful note creation.

    Formats a user-friendly message indicating successful note creation.

    Args:
        logger: Logger instance to use for logging
        notes_created: Number of notes successfully created

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_note_creation_success(logger, 10)
        # Output: "✓ Created 10 notes" or "[OK] Created 10 notes"
    """
    message = f"Created {notes_created} notes"
    formatted_message = _format_with_emoji(message, "✓", "[OK]")
    logger.info(formatted_message)


def log_tagging(logger: logging.Logger, tag_name: str) -> None:
    """Log tag addition.

    Formats a user-friendly message indicating tag addition.

    Args:
        logger: Logger instance to use for logging
        tag_name: Name of the tag being added

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_tagging(logger, "docai-processed")
        # Output: "🏷️ Added tag: docai-processed" or "[TAG] Added tag: docai-processed"
    """
    message = f"Added tag: {tag_name}"
    formatted_message = _format_with_emoji(message, "🏷️", "[TAG]")
    logger.info(formatted_message)


def log_disk_save(logger: logging.Logger, path: str) -> None:
    """Log disk save operation.

    Formats a user-friendly message indicating disk storage.

    Args:
        logger: Logger instance to use for logging
        path: Path where data was saved

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_disk_save(logger, "/path/to/save")
        # Output: "💾 Saved to disk: /path/to/save" or "[SAVE] Saved to disk:
        # /path/to/save"
    """
    message = f"Saved to disk: {path}"
    formatted_message = _format_with_emoji(message, "💾", "[SAVE]")
    logger.info(formatted_message)


def log_completion(logger: logging.Logger) -> None:
    """Log pipeline completion.

    Formats a user-friendly message indicating pipeline completion.

    Args:
        logger: Logger instance to use for logging

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_completion(logger)
        # Output: "✅ Pipeline completed" or "[DONE] Pipeline completed"
    """
    message = "Pipeline completed"
    formatted_message = _format_with_emoji(message, "✅", "[DONE]")
    logger.info(formatted_message)


def log_error(logger: logging.Logger, error: Exception, context: dict) -> None:
    """Log an error with structured context information.

    Formats a detailed error message including the exception details and
    relevant context (item_key, step, etc.) for debugging.

    Args:
        logger: Logger instance to use for logging
        error: Exception that was raised
        context: Dictionary containing context information such as:
            - item_key: Zotero item key
            - item_title: Title of the item being processed
            - step: Processing step where error occurred

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> context = {"item_key": "ABC123", "item_title": "My Doc", "step": "OCR"}
        >>> log_error(logger, ValueError("Invalid format"), context)
        # Output: "❌ Error processing \"My Doc\" (ABC123)\\n   Step: OCR\\n
        # Error: ValueError: Invalid format"
    """
    item_title = context.get("item_title", "Unknown")
    item_key = context.get("item_key", "Unknown")
    step = context.get("step", "Unknown")
    error_type = type(error).__name__
    error_message = str(error)

    if _supports_unicode():
        header = f'❌ Error processing "{item_title}" ({item_key})'
    else:
        header = f'[ERROR] Error processing "{item_title}" ({item_key})'

    message = f"{header}\n   Step: {step}\n   Error: {error_type}: {error_message}"

    logger.error(message)
    # Include full traceback only when in an active exception context
    if sys.exc_info()[0] is not None:
        logger.exception("Full traceback:")
    else:
        logger.error("(No active exception traceback available)")


def log_summary_table(
    logger: logging.Logger, results: list[ProcessingResult], skipped_count: int
) -> None:
    """Log a detailed per-item summary table.

    Creates a formatted table showing per-item processing results including
    item titles (truncated), pages extracted, and notes created. Includes
    totals row and skipped items count if applicable.

    Args:
        logger: Logger instance to use for logging
        results: List of ProcessingResult objects from pipeline execution
        skipped_count: Number of items skipped during discovery

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> results = [ProcessingResult(...), ...]
        >>> log_summary_table(logger, results, 2)
        # Output: Formatted table with per-item details
    """
    # Build table data from successful results
    table_data = []
    total_pages = 0
    total_notes = 0

    for result in results:
        if result.success:
            # Truncate title to 40 characters
            title = (
                result.item_title[:40] + "..."
                if len(result.item_title) > 40
                else result.item_title
            )
            table_data.append([title, result.pages_extracted, result.notes_created])
            total_pages += result.pages_extracted
            total_notes += result.notes_created

    # Add totals row if we have data (tabulate will add separator automatically)
    if table_data:
        table_data.append(["Total", total_pages, total_notes])

    # Choose table format based on unicode support
    tablefmt = "grid" if _supports_unicode() else "simple"

    # Log the table
    if table_data:
        headers = ["Item", "Pages", "Notes"]
        table_str = tabulate(table_data, headers=headers, tablefmt=tablefmt)
        logger.info("")
        logger.info("Summary:")
        logger.info(table_str)
    else:
        logger.info("Summary: No successful items to display")

    # Log skipped items if any
    if skipped_count > 0:
        message = f"Skipped: {skipped_count} items (already processed)"
        formatted_message = _format_with_emoji(message, "⏭️", "[SKIP]")
        logger.info("")
        logger.info(formatted_message)

    # Log error count if any failures
    failed_count = sum(1 for result in results if not result.success)
    if failed_count > 0:
        logger.info("")
        logger.info(f"Failed items: {failed_count}")


def log_timing_summary(logger: logging.Logger, total_time: float) -> None:
    """Log total execution time in a human-readable format.

    Formats the total execution time as minutes and seconds (e.g., "3m 15s")
    and logs it with appropriate emoji or fallback text.

    Args:
        logger: Logger instance to use for logging
        total_time: Total execution time in seconds

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> log_timing_summary(logger, 195.5)
        # Output: "⏱️ Total time: 3m 15s" or "[TIME] Total time: 3m 15s"
    """
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    message = f"Total time: {time_str}"
    formatted_message = _format_with_emoji(message, "⏱️", "[TIME]")
    logger.info("")
    logger.info(formatted_message)


def get_error_suggestion(error_message: str) -> str:
    """Get actionable suggestion based on error message pattern.

    Analyzes error messages to provide context-specific suggestions for
    resolving common issues encountered during pipeline execution.

    Args:
        error_message: Error message string to analyze

    Returns:
        Actionable suggestion string based on error pattern

    Example:
        >>> get_error_suggestion("Mistral API rate limit exceeded (429)")
        "Wait 60 seconds and retry, or reduce batch size"
    """
    error_lower = error_message.lower()

    if "rate limit" in error_lower or "429" in error_lower:
        return "Wait 60 seconds and retry, or reduce batch size"
    elif (
        "network" in error_lower
        or "connection" in error_lower
        or "timeout" in error_lower
    ):
        return "Check internet connection and retry"
    elif "not found" in error_lower or "404" in error_lower:
        return "Verify the item/attachment exists in Zotero"
    elif (
        "authentication" in error_lower or "401" in error_lower or "403" in error_lower
    ):
        return "Check API key validity and permissions"
    elif "upload" in error_lower:
        return "Check PDF file integrity and size limits"
    elif "ocr" in error_lower:
        return "Verify PDF is not corrupted or password-protected"
    else:
        return "Review error details and check logs for more information"


def log_error_summary(logger: logging.Logger, results: list[ProcessingResult]) -> None:
    """Log detailed error summary with suggestions for failed items.

    Filters results to find failed items and logs detailed error information
    including item title, key, error messages, and actionable suggestions
    for each error. Includes retry instructions at the end.

    Args:
        logger: Logger instance to use for logging
        results: List of ProcessingResult objects from pipeline execution

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> results = [ProcessingResult(success=False, errors=["..."]), ...]
        >>> log_error_summary(logger, results)
        # Output: Detailed error information with suggestions
    """
    # Filter to get only failed items
    failed_results = [result for result in results if not result.success]

    if not failed_results:
        return

    # Log header
    failed_count = len(failed_results)
    if _supports_unicode():
        header = f"❌ Errors ({failed_count} items failed):"
    else:
        header = f"[ERRORS] Errors ({failed_count} items failed):"

    logger.info("")
    logger.info(header)
    logger.info("")

    # Log each failed item with details
    for idx, result in enumerate(failed_results, start=1):
        logger.info(f'{idx}. "{result.item_title}" ({result.item_key})')

        for error in result.errors:
            logger.info(f"   Error: {error}")
            suggestion = get_error_suggestion(error)
            logger.info(f"   → Suggestion: {suggestion}")

        logger.info("")

    # Log retry instructions
    logger.info("To retry failed items:")
    logger.info("  1. Remove 'docai-error' tag from failed items in Zotero")
    logger.info("  2. Re-run the pipeline")
