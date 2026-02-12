"""
Zotero API client wrapper for domain-specific operations.

This module provides a high-level interface to the Zotero API using PyZotero,
with domain-specific methods for tag-based item retrieval, PDF downloads,
batch note creation, and tag management. All methods include comprehensive
error handling and logging for production use.
"""

import html
import logging
import time
from typing import Any, cast
from urllib.error import HTTPError, URLError

from pyzotero import zotero_errors
from pyzotero.zotero import Zotero

from zotero_docai_pipeline.clients.exceptions import (
    ZoteroAPIError,
    ZoteroAuthError,
    ZoteroItemNotFoundError,
)
from zotero_docai_pipeline.domain.config import ZoteroConfig
from zotero_docai_pipeline.domain.markdown_converter import convert_markdown_to_html
from zotero_docai_pipeline.domain.models import NotePayload

logger = logging.getLogger(__name__)


class ZoteroClient:
    """Client for interacting with the Zotero API.

    This class wraps PyZotero's Zotero class to provide domain-specific methods
    for the Zotero Document AI pipeline, including tag-based item retrieval,
    PDF downloads, batch note creation, and tag management.

    Example:
        >>> from zotero_docai_pipeline.domain.config import ZoteroConfig
        >>> from zotero_docai_pipeline.clients.zotero_client import ZoteroClient
        >>>
        >>> config = ZoteroConfig(
        ...     library_id="123456",
        ...     api_key="your_api_key_here"
        ... )
        >>> client = ZoteroClient(config)
        >>> items = client.get_items_by_tag("docai", "docai-processed")
    """

    @staticmethod
    def _format_note_identifier(note: NotePayload) -> str:
        """Format note identifier for logging.

        Args:
            note: NotePayload object

        Returns:
            Formatted string like "filename - Page N" or "filename (All Pages)"
        """
        if note.page_number is not None:
            return f"{note.pdf_filename} - Page {note.page_number}"
        else:
            return f"{note.pdf_filename} (All Pages)"

    def __init__(self, config: ZoteroConfig) -> None:
        """Initialize the Zotero client.

        Args:
            config: Zotero configuration containing library_id and api_key.
        """
        self.config = config
        self._zotero = Zotero(config.library_id, "user", config.api_key)
        logger.info(f"Initialized ZoteroClient for library_id: {config.library_id}")

    def get_items_by_tag(
        self, tag: str, exclude_tag: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve items by tag and filter out items with exclude_tag.

        Fetches all items with the specified tag, then filters out any items
        that also have the exclude_tag. For each item, retrieves child attachments
        to identify PDF files.

        Args:
            tag: Tag to filter items by. Items must have this tag to be included.
            exclude_tag: Optional tag to exclude. Items with this tag will be
                filtered out even if they have the target tag.

        Returns:
            List of dictionaries containing item metadata:
            - key: Zotero item key
            - title: Item title
            - tags: List of tag strings
            - attachments: List of attachment dicts with 'key' and 'filename'

        Raises:
            ZoteroAPIError: If API communication fails.
            ZoteroAuthError: If authentication fails.
        """
        try:
            logger.debug(f"Fetching items with tag: {tag}")
            items = self._zotero.items(tag=tag)

            # Filter out items with exclude_tag
            if exclude_tag:
                filtered_items = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_data_raw = item.get("data", {})
                    if not isinstance(item_data_raw, dict):
                        continue
                    item_tags = [
                        t.get("tag", "") for t in item_data_raw.get("tags", [])
                    ]
                    if exclude_tag not in item_tags:
                        filtered_items.append(item)
                items = filtered_items
                logger.debug(
                    f"Filtered to {len(items)} items after excluding tag: {exclude_tag}"
                )

            # Enrich items with attachment information
            result = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_data_raw = item.get("data", {})
                if not isinstance(item_data_raw, dict):
                    continue
                item_data = cast(dict[str, Any], item_data_raw)
                item_key = item.get("key")
                item_title = item_data.get("title", "Untitled")
                item_tags = [t.get("tag", "") for t in item_data.get("tags", [])]

                # Get child items (attachments)
                try:
                    children = self._zotero.children(item_key)
                    pdf_attachments = []
                    for child in children:
                        if not isinstance(child, dict):
                            continue
                        child_data_raw = child.get("data", {})
                        if not isinstance(child_data_raw, dict):
                            continue
                        child_data = cast(dict[str, Any], child_data_raw)
                        if (
                            child_data.get("itemType") == "attachment"
                            and child_data.get("contentType") == "application/pdf"
                        ):
                            pdf_attachments.append(
                                {
                                    "key": child.get("key"),
                                    "filename": child_data.get(
                                        "filename", "unknown.pdf"
                                    ),
                                }
                            )
                except Exception as e:
                    logger.warning(f"Failed to fetch children for item {item_key}: {e}")
                    pdf_attachments = []

                result.append(
                    {
                        "key": item_key,
                        "title": item_title,
                        "tags": item_tags,
                        "attachments": pdf_attachments,
                    }
                )

            logger.info(f"Retrieved {len(result)} items with tag '{tag}'")
            return result

        except HTTPError as e:
            if e.code in (401, 403):
                error_msg = (
                    f"Authentication failed while fetching items with tag '{tag}'"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroAuthError(error_msg, e) from e
            else:
                error_msg = (
                    f"API error while fetching items with tag '{tag}': HTTP {e.code}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroAPIError(error_msg, e) from e
        except URLError as e:
            error_msg = f"Network error while fetching items with tag '{tag}'"
            logger.error(f"{error_msg}: {e}")
            raise ZoteroAPIError(error_msg, e) from e
        except Exception as e:
            error_msg = f"Unexpected error while fetching items with tag '{tag}'"
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise ZoteroAPIError(error_msg, e) from e

    def download_pdf(self, item_key: str, attachment_key: str) -> bytes:
        """Download a PDF attachment from Zotero.

        Retrieves the binary content of a PDF attachment associated with
        a Zotero item.

        Args:
            item_key: Zotero item key that owns the attachment.
            attachment_key: Zotero attachment key for the PDF file.

        Returns:
            Binary PDF content as bytes.

        Raises:
            ZoteroItemNotFoundError: If the attachment doesn't exist (404).
            ZoteroAPIError: If API communication fails.
            ZoteroAuthError: If authentication fails.
        """
        try:
            logger.debug(
                f"Downloading PDF: item_key={item_key}, attachment_key={attachment_key}"
            )
            pdf_bytes_raw = self._zotero.file(attachment_key)
            # Ensure return type is bytes
            if isinstance(pdf_bytes_raw, bytes):
                pdf_bytes = pdf_bytes_raw
            elif isinstance(pdf_bytes_raw, str):
                pdf_bytes = pdf_bytes_raw.encode("utf-8")
            else:
                pdf_bytes = bytes(pdf_bytes_raw)
            logger.info(f"Successfully downloaded PDF: attachment_key={attachment_key}")
            return pdf_bytes
        except HTTPError as e:
            if e.code == 404:
                error_msg = (
                    f"PDF attachment not found: attachment_key={attachment_key}, "
                    f"item_key={item_key}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroItemNotFoundError(error_msg, e) from e
            elif e.code in (401, 403):
                error_msg = (
                    f"Authentication failed while downloading PDF: "
                    f"attachment_key={attachment_key}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroAuthError(error_msg, e) from e
            else:
                error_msg = (
                    f"API error while downloading PDF: "
                    f"attachment_key={attachment_key}, HTTP {e.code}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroAPIError(error_msg, e) from e
        except URLError as e:
            error_msg = (
                f"Network error while downloading PDF: attachment_key={attachment_key}"
            )
            logger.error(f"{error_msg}: {e}")
            raise ZoteroAPIError(error_msg, e) from e
        except Exception as e:
            error_msg = (
                f"Unexpected error while downloading PDF: "
                f"attachment_key={attachment_key}"
            )
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise ZoteroAPIError(error_msg, e) from e

    def create_notes_batch(
        self, notes: list[NotePayload], parent_key: str
    ) -> list[str]:
        """Create multiple notes in Zotero as a batch operation.

        Transforms NotePayload objects into Zotero note format and creates
        them as children of the specified parent item. Markdown content from
        NotePayload is automatically converted to HTML for Zotero's rich text
        editor (TinyMCE). If conversion fails, content is HTML-escaped and
        wrapped in <pre> tags as a fallback. This is an all-or-nothing
        operation: if any note fails to create, the entire batch fails.

        Args:
            notes: List of NotePayload objects to create as notes. Each note
                will be created as a child item of the parent. Markdown content
                from the payload is automatically converted to HTML for rich text
                display in Zotero.
            parent_key: Zotero item key that will be the parent of all notes.

        Returns:
            List of note keys for the successfully created notes.

        Raises:
            ValueError: If batch size exceeds 50 (Zotero API limit).
            ZoteroAPIError: If API call fails.
            ZoteroAuthError: If authentication fails.
        """
        if len(notes) > 50:
            error_msg = f"Batch size {len(notes)} exceeds Zotero API limit of 50"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Transform NotePayload objects to Zotero note dicts
            note_dicts = []
            for note in notes:
                # Convert markdown content to HTML for Zotero's rich text editor
                try:
                    html_content = convert_markdown_to_html(note.content)

                    # Fallback to basic HTML escaping if conversion produces
                    # empty result
                    if not html_content or not html_content.strip():
                        logger.warning(
                            f"Markdown conversion produced empty HTML for note "
                            f"'{ZoteroClient._format_note_identifier(note)}', "
                            f"falling back to HTML escaping"
                        )
                        html_content = f"<pre>{html.escape(note.content)}</pre>"

                except Exception as e:
                    # Fallback to basic HTML escaping on conversion failure
                    logger.warning(
                        f"Failed to convert markdown to HTML for note "
                        f"'{ZoteroClient._format_note_identifier(note)}': {e}. "
                        f"Falling back to HTML escaping"
                    )
                    html_content = f"<pre>{html.escape(note.content)}</pre>"

                note_dict = {
                    "itemType": "note",
                    "note": html_content,  # Use converted HTML instead of raw markdown
                    "parentItem": parent_key,
                    "tags": [],
                }
                note_dicts.append(note_dict)

            logger.debug(
                f"Creating batch of {len(note_dicts)} notes for parent_key={parent_key}"
            )
            result = self._zotero.create_items(note_dicts, parentid=parent_key)

            # Check for partial failures in batch operation
            failed = result.get("failed", {})
            if failed:
                # Extract successful note keys for rollback
                successful = result.get("successful", {})
                successful_keys = []

                if isinstance(successful, dict):
                    for idx in successful:
                        created_item = successful[idx]
                        if isinstance(created_item, dict) and "key" in created_item:
                            successful_keys.append(created_item["key"])
                        elif isinstance(created_item, list) and len(created_item) > 0:
                            item_data = (
                                created_item[0]
                                if isinstance(created_item[0], dict)
                                else created_item
                            )
                            if isinstance(item_data, dict) and "key" in item_data:
                                successful_keys.append(item_data["key"])

                # Log successful keys before rollback
                if successful_keys:
                    successful_titles = [
                        ZoteroClient._format_note_identifier(notes[int(idx)])
                        for idx in successful
                        if isinstance(idx, (str, int)) and int(idx) < len(notes)
                    ]
                    logger.warning(
                        f"Rolling back {len(successful_keys)} successfully "
                        f"created notes: {', '.join(successful_titles)}"
                    )

                    # Attempt rollback
                    try:
                        self.delete_notes(successful_keys)
                    except Exception as rollback_error:
                        logger.error(
                            f"Rollback failed for {len(successful_keys)} notes: "
                            f"{rollback_error}",
                            exc_info=True,
                        )

                # Map failed indices to note titles for error message
                note_titles = [
                    ZoteroClient._format_note_identifier(note) for note in notes
                ]
                failed_titles = []
                for idx in failed:
                    try:
                        idx_int = int(idx) if isinstance(idx, str) else idx
                        if 0 <= idx_int < len(note_titles):
                            failed_titles.append(note_titles[idx_int])
                    except (ValueError, TypeError):
                        continue

                error_msg = (
                    f"Batch note creation failed for {len(failed)} notes "
                    f"out of {len(note_dicts)}"
                )
                logger.error(f"{error_msg} for parent_key={parent_key}")
                logger.error(f"Failed note titles: {', '.join(failed_titles)}")
                raise ZoteroAPIError(error_msg)

            # Extract note keys from successful creation result
            # PyZotero's create_items returns a dict with 'successful' and 'failed' keys
            # 'successful' contains a dict mapping indices to created item data
            note_keys = []
            successful = result.get("successful", {})

            if isinstance(successful, dict):
                # Most common case: successful is a dict mapping indices to item data
                for idx in sorted(
                    successful.keys(),
                    key=lambda x: (
                        int(x)
                        if isinstance(x, str) and x.isdigit()
                        else (x if isinstance(x, int) else 0)
                    ),
                ):
                    created_item = successful[idx]
                    if isinstance(created_item, dict) and "key" in created_item:
                        note_keys.append(created_item["key"])
                    elif isinstance(created_item, list) and len(created_item) > 0:
                        # Handle case where item data is nested in a list
                        item_data = (
                            created_item[0]
                            if isinstance(created_item[0], dict)
                            else created_item
                        )
                        if isinstance(item_data, dict) and "key" in item_data:
                            note_keys.append(item_data["key"])
            elif isinstance(successful, list):
                # Fallback: successful is a list of created items
                for item in successful:
                    if isinstance(item, dict) and "key" in item:
                        note_keys.append(item["key"])

            # Validate that we extracted the expected number of keys
            if len(note_keys) != len(note_dicts):
                logger.warning(
                    f"Extracted {len(note_keys)} note keys but expected "
                    f"{len(note_dicts)}. Result structure: "
                    f"{type(result).__name__}, successful type: "
                    f"{type(successful).__name__}"
                )

            note_titles = [ZoteroClient._format_note_identifier(note) for note in notes]
            logger.info(
                f"Successfully created {len(note_dicts)} notes for "
                f"parent_key={parent_key}"
            )
            logger.debug(f"Note titles: {', '.join(note_titles)}")
            logger.debug(f"Created note keys: {note_keys}")

            return note_keys

        except HTTPError as e:
            note_titles = [ZoteroClient._format_note_identifier(note) for note in notes]
            if e.code in (401, 403):
                error_msg = (
                    f"Authentication failed while creating notes batch for "
                    f"parent_key={parent_key}"
                )
                logger.error(f"{error_msg}: {e}")
                logger.error(f"Failed note titles: {', '.join(note_titles)}")
                raise ZoteroAuthError(error_msg, e) from e
            else:
                error_msg = (
                    f"API error while creating notes batch for "
                    f"parent_key={parent_key}: HTTP {e.code}"
                )
                logger.error(f"{error_msg}: {e}")
                logger.error(f"Failed note titles: {', '.join(note_titles)}")
                raise ZoteroAPIError(error_msg, e) from e
        except URLError as e:
            note_titles = [ZoteroClient._format_note_identifier(note) for note in notes]
            error_msg = (
                f"Network error while creating notes batch for parent_key={parent_key}"
            )
            logger.error(f"{error_msg}: {e}")
            logger.error(f"Failed note titles: {', '.join(note_titles)}")
            raise ZoteroAPIError(error_msg, e) from e
        except Exception as e:
            note_titles = [ZoteroClient._format_note_identifier(note) for note in notes]
            error_msg = (
                f"Unexpected error while creating notes batch for "
                f"parent_key={parent_key}"
            )
            logger.error(f"{error_msg}: {e}", exc_info=True)
            logger.error(f"Failed note titles: {', '.join(note_titles)}")
            raise ZoteroAPIError(error_msg, e) from e

    def delete_notes(self, note_keys: list[str]) -> None:
        """Delete multiple notes from Zotero by their keys.

        This method is used for rollback operations when batch note creation
        fails partway through. It attempts to delete all specified notes,
        logging warnings for any that fail to delete (but not raising exceptions
        to allow cleanup to continue).

        Args:
            note_keys: List of Zotero note keys to delete.

        Raises:
            ZoteroAPIError: If API communication fails (but individual note
                deletion failures are logged as warnings, not raised).
        """
        if not note_keys:
            return

        logger.debug(f"Deleting {len(note_keys)} notes for rollback")
        deleted_count = 0
        failed_keys = []

        for note_key in note_keys:
            try:
                self._zotero.delete_item(note_key)
                deleted_count += 1
                logger.debug(f"Deleted note: {note_key}")
            except HTTPError as e:
                if e.code == 404:
                    # Note already deleted or doesn't exist - not an error for rollback
                    logger.debug(f"Note {note_key} not found (may already be deleted)")
                    deleted_count += 1
                elif e.code in (401, 403):
                    error_msg = f"Authentication failed while deleting note {note_key}"
                    logger.warning(f"{error_msg}: {e}")
                    failed_keys.append(note_key)
                else:
                    error_msg = (
                        f"API error while deleting note {note_key}: HTTP {e.code}"
                    )
                    logger.warning(f"{error_msg}: {e}")
                    failed_keys.append(note_key)
            except URLError as e:
                error_msg = f"Network error while deleting note {note_key}"
                logger.warning(f"{error_msg}: {e}")
                failed_keys.append(note_key)
            except Exception as e:
                error_msg = f"Unexpected error while deleting note {note_key}"
                logger.warning(f"{error_msg}: {e}", exc_info=True)
                failed_keys.append(note_key)

        if failed_keys:
            logger.warning(
                f"Failed to delete {len(failed_keys)} notes during rollback: "
                f"{failed_keys}"
            )
        else:
            logger.info(f"Successfully deleted {deleted_count} notes during rollback")

    def add_tag(self, item_key: str, tag: str) -> None:
        """Add a tag to a Zotero item.

        Fetches the item, adds the specified tag, and updates it via the API.

        Args:
            item_key: Zotero item key to add the tag to.
            tag: Tag name to add.

        Raises:
            ZoteroAPIError: If API call fails.
            ZoteroAuthError: If authentication fails.
            ZoteroItemNotFoundError: If the item doesn't exist (404).
        """
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"Adding tag '{tag}' to item_key={item_key} (attempt {attempt})"
                )

                # Fetch item immediately before update to get latest version
                item = self._zotero.item(item_key)

                # Check if tag already exists
                if not isinstance(item, dict):
                    raise ZoteroAPIError(
                        f"Invalid item structure for item_key={item_key}"
                    )
                item_data_raw = item.get("data", {})
                if not isinstance(item_data_raw, dict):
                    raise ZoteroAPIError(
                        f"Invalid item data structure for item_key={item_key}"
                    )
                item_data = cast(dict[str, Any], item_data_raw)
                item_tags = [
                    t.get("tag", "") if isinstance(t, dict) else t
                    for t in item_data.get("tags", [])
                ]
                if tag in item_tags:
                    logger.info(
                        f"Tag '{tag}' already exists on item_key={item_key}, "
                        f"skipping update"
                    )
                    return

                # Add tag to the item (modifies item in-place, preserves version)
                # Don't re-fetch - update immediately with the version we have
                # If version changed, the 412 error will trigger a retry with
                # fresh fetch
                self._zotero.add_tags(item, tag)

                # Update immediately - if version changed, we'll get 412 and retry
                self._zotero.update_item(item)
                logger.info(f"Successfully added tag '{tag}' to item_key={item_key}")
                return

            except zotero_errors.PreConditionFailedError as e:
                # Even if we got 412, the tag might have been added anyway
                # Verify by fetching the item and checking if tag exists
                try:
                    time.sleep(0.1)  # Small delay to allow Zotero to process
                    verify_item = self._zotero.item(item_key)
                    if not isinstance(verify_item, dict):
                        raise ZoteroAPIError(
                            f"Invalid item structure for item_key={item_key}"
                        )
                    verify_item_data_raw = verify_item.get("data", {})
                    if not isinstance(verify_item_data_raw, dict):
                        raise ZoteroAPIError(
                            f"Invalid item data structure for item_key={item_key}"
                        )
                    verify_item_data = cast(dict[str, Any], verify_item_data_raw)
                    verify_tags = [
                        t.get("tag", "") if isinstance(t, dict) else t
                        for t in verify_item_data.get("tags", [])
                    ]

                    if tag in verify_tags:
                        # Tag was added despite 412 error - consider it success
                        logger.info(
                            f"Tag '{tag}' was successfully added to "
                            f"item_key={item_key} despite version conflict "
                            f"(attempt {attempt})"
                        )
                        return
                except Exception as verify_error:
                    # Verification failed, but continue with retry logic
                    logger.debug(
                        f"Failed to verify tag after 412 error: {verify_error}"
                    )

                if attempt == max_retries:
                    # Final attempt failed, raise error
                    error_msg = (
                        f"Failed to add tag '{tag}' to item_key={item_key} "
                        f"after {max_retries} retry attempts"
                    )
                    logger.error(f"{error_msg}: {e}")
                    raise ZoteroAPIError(error_msg, e) from e

                # Exponential backoff before retry (longer delays to allow
                # version to stabilize)
                delay = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1.0s, 2.0s, 4.0s
                logger.debug(
                    f"Version conflict on attempt {attempt}, "
                    f"waiting {delay}s before retry"
                )
                time.sleep(delay)
                continue  # Retry the loop from the beginning (fresh fetch)

            except HTTPError as e:
                # Non-retryable HTTP errors - break out and raise immediately
                if e.code == 404:
                    error_msg = (
                        f"Item not found while adding tag: item_key={item_key}, "
                        f"tag={tag}"
                    )
                    logger.error(f"{error_msg}: {e}")
                    raise ZoteroItemNotFoundError(error_msg, e) from e
                elif e.code in (401, 403):
                    error_msg = (
                        f"Authentication failed while adding tag '{tag}' to "
                        f"item_key={item_key}"
                    )
                    logger.error(f"{error_msg}: {e}")
                    raise ZoteroAuthError(error_msg, e) from e
                else:
                    error_msg = (
                        f"API error while adding tag '{tag}' to "
                        f"item_key={item_key}: HTTP {e.code}"
                    )
                    logger.error(f"{error_msg}: {e}")
                    raise ZoteroAPIError(error_msg, e) from e
            except URLError as e:
                # Network errors - break out and raise immediately
                error_msg = (
                    f"Network error while adding tag '{tag}' to item_key={item_key}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroAPIError(error_msg, e) from e
            except Exception as e:
                # Other unexpected errors - break out and raise immediately
                error_msg = (
                    f"Unexpected error while adding tag '{tag}' to item_key={item_key}"
                )
                logger.error(f"{error_msg}: {e}", exc_info=True)
                raise ZoteroAPIError(error_msg, e) from e

    def remove_tag(self, item_key: str, tag: str) -> None:
        """Remove a tag from a Zotero item.

        Fetches the item, removes the specified tag from its tags list,
        and updates it via the API.

        Args:
            item_key: Zotero item key to remove the tag from.
            tag: Tag name to remove.

        Raises:
            ZoteroAPIError: If API call fails.
            ZoteroAuthError: If authentication fails.
            ZoteroItemNotFoundError: If the item doesn't exist (404).
        """
        try:
            logger.debug(f"Removing tag '{tag}' from item_key={item_key}")
            item = self._zotero.item(item_key)

            # Manually remove tag from item['data']['tags'] list
            if not isinstance(item, dict):
                raise ZoteroAPIError(f"Invalid item structure for item_key={item_key}")
            item_data_raw = item.get("data", {})
            if not isinstance(item_data_raw, dict):
                raise ZoteroAPIError(
                    f"Invalid item data structure for item_key={item_key}"
                )
            item_data = cast(dict[str, Any], item_data_raw)
            tags = item_data.get("tags", [])
            item_data["tags"] = [t for t in tags if t.get("tag") != tag]

            self._zotero.update_item(item)
            logger.info(f"Successfully removed tag '{tag}' from item_key={item_key}")
        except zotero_errors.PreConditionFailedError as e:
            # Handle version conflict: retry with exponential backoff (up to 3 attempts)
            max_retries = 3
            retry_delay = 0.2  # 200ms initial delay
            last_error = e

            for attempt in range(1, max_retries + 1):
                logger.debug(
                    f"Version conflict detected for item_key={item_key}, "
                    f"retry attempt {attempt}/{max_retries}"
                )
                try:
                    # Small delay before retry to allow concurrent operations
                    # to complete
                    if attempt > 1:
                        time.sleep(
                            retry_delay * attempt
                        )  # Exponential backoff: 0.2s, 0.4s, 0.6s

                    # Re-fetch item to get latest version
                    item = self._zotero.item(item_key)

                    # Check if tag already removed (may have been removed by
                    # another process)
                    if not isinstance(item, dict):
                        raise ZoteroAPIError(
                            f"Invalid item structure for item_key={item_key}"
                        )
                    item_data_raw = item.get("data", {})
                    if not isinstance(item_data_raw, dict):
                        raise ZoteroAPIError(
                            f"Invalid item data structure for item_key={item_key}"
                        )
                    item_data = cast(dict[str, Any], item_data_raw)
                    item_tags = [
                        t.get("tag", "") if isinstance(t, dict) else t
                        for t in item_data.get("tags", [])
                    ]
                    if tag not in item_tags:
                        logger.info(
                            f"Tag '{tag}' already removed from "
                            f"item_key={item_key}, skipping update"
                        )
                        return

                    # Manually remove tag from item['data']['tags'] list
                    # item is already verified as dict above, but reuse
                    # item_data_raw from above check to avoid redundant access
                    item_data = cast(dict[str, Any], item_data_raw)
                    tags = item_data.get("tags", [])
                    item_data["tags"] = [t for t in tags if t.get("tag") != tag]

                    # Update with fresh version
                    self._zotero.update_item(item)
                    logger.info(
                        f"Successfully removed tag '{tag}' from "
                        f"item_key={item_key} after version conflict retry "
                        f"(attempt {attempt})"
                    )
                    return
                except zotero_errors.PreConditionFailedError as retry_error:
                    last_error = retry_error
                    if attempt == max_retries:
                        # Final attempt failed, raise error
                        error_msg = (
                            f"Failed to remove tag '{tag}' from "
                            f"item_key={item_key} after {max_retries} retry "
                            f"attempts"
                        )
                        logger.error(f"{error_msg}: {last_error}")
                        raise ZoteroAPIError(error_msg, last_error) from last_error
                    # Continue to next retry attempt
                    continue
                except Exception as retry_error:
                    error_msg = (
                        f"Failed to remove tag '{tag}' from "
                        f"item_key={item_key} after version conflict retry "
                        f"(attempt {attempt})"
                    )
                    logger.error(f"{error_msg}: {retry_error}")
                    raise ZoteroAPIError(error_msg, retry_error) from retry_error
        except HTTPError as e:
            if e.code == 404:
                error_msg = (
                    f"Item not found while removing tag: item_key={item_key}, tag={tag}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroItemNotFoundError(error_msg, e) from e
            elif e.code in (401, 403):
                error_msg = (
                    f"Authentication failed while removing tag '{tag}' from "
                    f"item_key={item_key}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroAuthError(error_msg, e) from e
            else:
                error_msg = (
                    f"API error while removing tag '{tag}' from "
                    f"item_key={item_key}: HTTP {e.code}"
                )
                logger.error(f"{error_msg}: {e}")
                raise ZoteroAPIError(error_msg, e) from e
        except URLError as e:
            error_msg = (
                f"Network error while removing tag '{tag}' from item_key={item_key}"
            )
            logger.error(f"{error_msg}: {e}")
            raise ZoteroAPIError(error_msg, e) from e
        except Exception as e:
            error_msg = (
                f"Unexpected error while removing tag '{tag}' from item_key={item_key}"
            )
            logger.error(f"{error_msg}: {e}", exc_info=True)
            raise ZoteroAPIError(error_msg, e) from e
