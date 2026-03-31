# zotero-docai-pipeline

![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Automate PDF-to-Markdown extraction for Zotero attachments using OCR providers (PageIndex or Mistral). Results are saved as rich text Zotero notes ready for Notero/Notion sync, with optional hierarchical tree structure extraction.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Tag Adding](#tag-adding)
- [PDF Download](#pdf-download)
- [Extraction Modes](#extraction-modes)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

## Quick Start

Complete the [Installation](#installation) steps first, then follow this workflow:

1. **Tag items in Zotero:**
   - Tag items you want to process with `docai` tag

2. **Run the pipeline:**
   ```bash
   python main.py ocr.enabled=true
   ```

3. **Verify results:**
   - Check your Zotero library for newly created notes
   - Successfully processed items are tagged with `docai-processed`

### Dry-Run Mode

Test your configuration without creating notes:
```bash
python main.py processing.dry_run=true ocr.enabled=true
```

## Installation

### Prerequisites

- Zotero API access (API key and library ID)
- API key for your chosen OCR provider:
  - PageIndex API key (for PageIndex OCR)
  - Mistral API key (for Mistral OCR)

### Install Dependencies

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate zotero-docai-pipeline
```

The `environment.yml` file includes all required dependencies: Python 3.8+, Hydra Core, PyZotero, Mistral AI SDK, PageIndex SDK, and markdown processing libraries.

### API Key Configuration

Set the required environment variables:

```bash
export ZOTERO_LIBRARY_ID="your-library-id"
export ZOTERO_API_KEY="your-zotero-api-key"
```

For OCR provider, set one of:
```bash
export PAGEINDEX_API_KEY="your-pageindex-api-key"  # For PageIndex OCR
# or
export MISTRAL_API_KEY="your-mistral-api-key"      # For Mistral OCR
```

**Provider Setup:**
- **PageIndex:** Obtain API key from [PageIndex dashboard](https://docs.pageindex.ai). Optional SDK mode: install with `pip install pageindex` and set `use_sdk: true` in `conf/ocr/pageindex.yaml`
- **Mistral:** Obtain API key from [Mistral platform](https://docs.mistral.ai)

## Configuration

The pipeline uses Hydra for configuration management. Most settings can be overridden via command-line arguments. This section covers essential options and command-line configuration.

### Essential Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ocr.provider` | string | `pageindex` | OCR provider selection (`mistral` or `pageindex`) |
| `ocr.enabled` | boolean | `false` | Enable OCR processing (required to create notes) |
| `download.enabled` | boolean | `false` | Enable PDF download from Zotero |
| `tag_adding.enabled` | boolean | `false` | Enable citationKey-based Zotero tag assignment |
| `tree_structure.enabled` | boolean | `true` | Enable hierarchical tree structure extraction |
| `processing.extraction_mode` | string | `all_at_once` | Note organization mode (`all_at_once` or `page_by_page`) |
| `processing.batch_size` | integer | varies | Batch processing size (provider-dependent) |
| `download.tag` | string | `docai` | Primary Zotero tag used by `Pipeline._discover_items()` to select items for download/tagging; falls back to `zotero.tags.input` if unset |
| `download.upload_folder` | string | `./downloads` | Local directory for downloaded PDFs |
| `download.max_concurrent_downloads` | integer | `5` | Maximum number of concurrent downloads |
| `processing.cleanup_uploaded_files` | boolean | `false` | Controls file deletion after processing (default: false = keep files) |

### Choosing an OCR Provider

The pipeline supports two OCR providers:

| Feature | PageIndex | Mistral |
|---------|-----------|---------|
| Processing Type | Asynchronous | Synchronous |
| Batch Processing | Optimized (FIFO queue) | Single document |
| Tree Structure | ✅ Full support | ⚠️ Requires PageIndex credentials |
| Best For | Batch processing, hierarchical organization | Single documents, fast processing |

**Recommendation:** For new installations, **PageIndex OCR** is recommended for batch processing and hierarchical document organization. Use **Mistral OCR** for single-document processing when tree structure is not needed.

### Command-Line Configuration

Override configuration from the command line:
```bash
# Change OCR provider
python main.py ocr=pageindex ocr.enabled=true

# Enable/disable tree extraction
python main.py tree_structure.enabled=true ocr.enabled=true

# Change extraction mode
python main.py processing.extraction_mode=page_by_page ocr.enabled=true

# Multiple overrides
python main.py ocr=pageindex tree_structure.enabled=true processing.dry_run=true ocr.enabled=true
```

## Tag Adding

Optional step that applies Zotero tags to items by matching their **citation keys** (no OCR/notes are created in tag-adding-only mode).

### Key configuration
- `tag_adding.enabled`: set to `true` to enable tag adding.
- `tag_adding.assignments`: a dictionary mapping `citation_key -> list[str]` (tags to add).
- `tag_adding.replace_all_existing_tags` (opt-in, destructive): when `true`, all existing tags are replaced with the assigned tags.

### Citation key matching
- Matching is **exact**, **case-sensitive**, and **whitespace-trimmed**.
- The citation key is resolved from Zotero data using this precedence:
  1. `item["data"]["citationKey"]` (preferred when present and non-empty)
  2. a `Citation Key: <key>` line in `item["data"]["extra"]`

### Environment override (recommended for large mappings)
- If you set `TAG_ADDING_ASSIGNMENTS_JSON` to a JSON object of the form `{ "citation_key": ["tag1", "tag2"] }`, the app will enable tag-adding automatically.

Example:
```bash
export TAG_ADDING_ASSIGNMENTS_JSON='{"Smith2020":["docai-tag1","docai-tag2"]}'
python main.py tag_adding.enabled=true
```

## PDF Download

Optional step that downloads PDFs from Zotero items to local disk (used as an input for OCR, and also supported in download-only mode).

### Key configuration
- `download.enabled`: set to `true` to enable PDF downloads.
- `download.tag`: primary Zotero tag that `Pipeline._discover_items()` uses to select items for downloading and processing (default: `docai`).
- `zotero.tags.input`: backward-compatible fallback tag used by `Pipeline._discover_items()` when `download.tag` is absent.
- `download.upload_folder`: local directory for downloaded PDFs (default: `./downloads`).
- `download.preserve_filenames`: whether to preserve the original PDF filenames (default: `true`).
- `download.create_subfolders`: whether to create subfolders under `upload_folder` (default: `false`).
- `download.skip_existing`: whether to skip PDFs that already exist locally (default: `true`).
- `download.max_concurrent_downloads`: maximum number of concurrent downloads (default: `5`).
- Retry is configurable via `download.retry.*` (see `conf/download/default.yaml`).

### Important constraint
- `processing.dry_run=true` cannot be combined with `download.enabled=true`.

Examples:
```bash
# Download-only
python main.py download.enabled=true

# Download + OCR
python main.py download.enabled=true ocr.enabled=true
```

## Extraction Modes

Choose how pages are organized into notes. Both modes work with all OCR providers. Use `all_at_once` (default) to consolidate all document content in a single note, or `page_by_page` for separate notes per page.

| Feature | All-at-Once (Default) | Page-by-Page |
|---------|----------------------|--------------|
| Notes per PDF | Single note | One note per page |
| Page separators | `--- Page N ---` markers | N/A (each page is separate) |
| Note title format | `"{filename} -\nOCR (All Pages)"` | `"Page N -\n{filename} -\nOCR"` |
| Best for | Single consolidated view, easier searching | Page-by-page organization, large documents |
| Size limits | Auto-splits if content exceeds threshold | Auto-splits per page if needed |

## Troubleshooting

Common issues and solutions when running the pipeline:

### API Key and Tree Extraction Issues

**API keys**: Ensure the correct environment variable is set (see [Installation](#installation)):
- `PAGEINDEX_API_KEY` for PageIndex OCR
- `MISTRAL_API_KEY` for Mistral OCR

**Tree extraction**: Requires PageIndex provider (`ocr.provider: pageindex`), valid `PAGEINDEX_API_KEY`, and `tree_structure.enabled: true` (see [Configuration](#configuration)).

### No Items Found

Verify:
- Items are tagged with the input tag (default: `docai`) - see [Quick Start](#quick-start)
- Items are not already tagged with the output tag (default: `docai-processed`)
- `zotero.library_id` is correctly configured
- Zotero API key has access to the specified library

### Note Size Exceeded

If notes exceed Zotero's size limits:
- Adjust `processing.note_size_threshold` to reduce note size
- Enable `processing.auto_split_oversized_notes` to automatically split large notes
- Consider using `extraction_mode: page_by_page` for very large documents (see [Extraction Modes](#extraction-modes))

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue on GitHub.

For major changes, please open an issue first to discuss what you would like to change.
