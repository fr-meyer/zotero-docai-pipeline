# zotero-docai-pipeline

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Automate PDF-to-Markdown extraction for Zotero attachments using OCR providers (PageIndex or Mistral). Results are saved as rich text Zotero notes ready for Notero/Notion sync, with optional hierarchical tree structure extraction.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
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
   python main.py
   ```

3. **Verify results:**
   - Check your Zotero library for newly created notes
   - Successfully processed items are tagged with `docai-processed`

### Dry-Run Mode

Test your configuration without creating notes:
```bash
python main.py processing.dry_run=true
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
| `tree_structure.enabled` | boolean | `true` | Enable hierarchical tree structure extraction |
| `processing.extraction_mode` | string | `all_at_once` | Note organization mode (`all_at_once` or `page_by_page`) |
| `processing.batch_size` | integer | varies | Batch processing size (provider-dependent) |
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
python main.py ocr=pageindex

# Enable/disable tree extraction
python main.py tree_structure.enabled=true

# Change extraction mode
python main.py processing.extraction_mode=page_by_page

# Multiple overrides
python main.py ocr=pageindex tree_structure.enabled=true processing.dry_run=true
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
