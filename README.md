# zotero-docai-pipeline

![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Automate PDF-to-Markdown extraction for Zotero attachments using OCR providers (PageIndex or Mistral). Results are saved as rich text Zotero notes ready for Notero/Notion sync, with optional hierarchical tree structure extraction.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Runtime Prerequisites](#runtime-prerequisites)
- [Configuration](#configuration)
- [Item Selection & Tagging](#item-selection--tagging)
- [Tag Adding](#tag-adding)
- [PDF Download](#pdf-download)
- [Extraction Modes](#extraction-modes)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

## Quick Start

Complete the [Installation](#installation) and [Runtime Prerequisites](#runtime-prerequisites) steps first, then follow this workflow:

1. **Tag items in Zotero:**
   - Tag items you want to process with the configured include tag (default: `docai`)

2. **Run the pipeline:**
   ```bash
   zotero-docai-pipeline ocr.enabled=true
   # or equivalently:
   python -m zotero_docai_pipeline ocr.enabled=true
   ```

3. **Verify results:**
   - Check your Zotero library for newly created notes
   - Successfully processed items are tagged with the configured success tag(s) (default: `docai-processed`)

### Dry-Run Mode

Test your configuration without creating notes:
```bash
zotero-docai-pipeline processing.dry_run=true ocr.enabled=true
# or equivalently:
python -m zotero_docai_pipeline processing.dry_run=true ocr.enabled=true
```

Dry-run mode performs the full item-selection logic but **does not write any tags or notes** to Zotero. For every matched item the following details are logged:

- **Title**
- **Citation key** — displayed as `[none]` when absent
- **DOI** — displayed as `[none]` when absent
- **Author summary** — e.g. `"Smith, Doe, and Lee"` or `[no authors]`
- **PDF attachment count**
- **Current tags** on the item
- **Would-apply tags** — the success tags that would be written on a real run

A summary line is printed at the end showing the total matched items, total PDFs, and total excluded items.

## Installation

### Install the package

```bash
pip install .        # standard install
pip install -e .     # editable / development install
```

After installation, `zotero-docai-pipeline` is available on `PATH` and can be verified from any directory:
```bash
zotero-docai-pipeline --help
```

### Optional: conda environment

If you prefer using conda for dependency isolation:
```bash
conda env create -f environment.yml
conda activate zotero-docai-pipeline
pip install .
```

The `environment.yml` file includes all required dependencies: Python 3.8+, Hydra Core, PyZotero, Mistral AI SDK, PageIndex SDK, and markdown processing libraries.

## Runtime Prerequisites

The following environment variables must be set **before running** the pipeline.

### Zotero credentials (required)

```bash
export ZOTERO_LIBRARY_ID="your-library-id"
export ZOTERO_API_KEY="your-zotero-api-key"
```

### OCR provider key (at least one required)

```bash
export PAGEINDEX_API_KEY="your-pageindex-api-key"  # For PageIndex OCR
# or
export MISTRAL_API_KEY="your-mistral-api-key"      # For Mistral OCR
```

**Provider Setup:**
- **PageIndex:** Obtain API key from [PageIndex dashboard](https://docs.pageindex.ai). Optional SDK mode: install with `pip install pageindex` and set `use_sdk: true` in `zotero_docai_pipeline/conf/ocr/pageindex.yaml`.
- **Mistral:** Obtain API key from [Mistral platform](https://docs.mistral.ai).

> **Note:** Missing or invalid environment variables may cause the CLI to error before help text is shown. If you see unexpected errors on startup, verify your environment variables are set correctly.

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
| `tagging.selection.include.values` | list | `[docai]` | Tags items must match to be selected for processing |
| `tagging.selection.include.operator` | string | `or` | How include tags are combined (`and` / `or`) |
| `tagging.selection.exclude.values` | list | `[docai-processed]` | Tags that disqualify items from selection |
| `tagging.selection.exclude.operator` | string | `or` | How exclude tags are combined (`and` / `or`) |
| `tagging.apply_on_success.values` | list | `[docai-processed]` | Tags added to items on successful processing |
| `tagging.apply_on_error.values` | list | `[docai-error]` | Tags added to items on failed processing |
| `tagging.include_abstract` | boolean | `false` | Include abstract in `paper_metadata` passed to OCR |
| `zotero.error_tagging_enabled` | boolean | `true` | Whether error tags are applied on failure |
| `download.upload_folder` | string | `./downloads` | Local directory for downloaded PDFs (must be set to an explicit path when `download.enabled=true`) |
| `download.max_concurrent_downloads` | integer | `5` | Maximum number of concurrent downloads |
| `storage.base_dir` | string | `./data/ocr_output` | Base directory for on-disk storage (must be set to an explicit path when `processing.save_to_disk=true`) |
| `processing.cleanup_uploaded_files` | boolean | `false` | Controls file deletion after processing (default: false = keep files) |

> **Explicit output paths required in path-consuming modes:**
> - `download.enabled=true` requires an explicit `download.upload_folder` override.
> - `processing.save_to_disk=true` requires an explicit `storage.base_dir` override.
>
> The packaged placeholder defaults are rejected; the CLI will exit with an actionable error if they are not overridden.

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
zotero-docai-pipeline ocr=pageindex ocr.enabled=true
# or: python -m zotero_docai_pipeline ocr=pageindex ocr.enabled=true

# Enable/disable tree extraction
zotero-docai-pipeline tree_structure.enabled=true ocr.enabled=true
# or: python -m zotero_docai_pipeline tree_structure.enabled=true ocr.enabled=true

# Change extraction mode
zotero-docai-pipeline processing.extraction_mode=page_by_page ocr.enabled=true
# or: python -m zotero_docai_pipeline processing.extraction_mode=page_by_page ocr.enabled=true

# Multiple overrides
zotero-docai-pipeline ocr=pageindex tree_structure.enabled=true processing.dry_run=true ocr.enabled=true
# or: python -m zotero_docai_pipeline ocr=pageindex tree_structure.enabled=true processing.dry_run=true ocr.enabled=true
```

## Item Selection & Tagging

The pipeline uses a flexible, config-driven tagging system to control which items are processed and how they are tagged afterward. All tagging behaviour is defined under the `tagging` config group.

### Item Selection (`tagging.selection`)

Items are selected for processing based on **include** and **exclude** tag rules:

- **`selection.include.values`** — a list of tags an item must carry to be eligible. Multiple tags are combined with the `include.operator` (`and` requires all tags; `or` requires at least one).
- **`selection.exclude.values`** — a list of tags that disqualify an item. Combined with the `exclude.operator` using the same logic.
- **`conflict_resolution`** controls what happens when an item matches both include and exclude rules:
  - **`exclude_wins`** *(default)* — the item is excluded and not processed.
  - **`include_wins`** — the item is included and processed normally.

### Success and Error Tagging

After processing, the pipeline applies outcome-based tags:

- **`apply_on_success.values`** — tags added to items that were processed successfully (default: `[docai-processed]`).
- **`apply_on_error.values`** — tags added to items that failed during processing (default: `[docai-error]`). Error tagging can be disabled globally by setting `zotero.error_tagging_enabled: false`.

### Rich Metadata

Setting `tagging.include_abstract: true` includes the item's abstract in `paper_metadata` passed to the OCR provider, which can improve extraction quality for academic papers.

### Processing Summary Output

Each processed item's entry in `processing_summary.json` contains a nested **`paper_metadata`** block with the following fields:

- `citation_key`, `title`, `doi`, `item_type`, `date`, `year`, `publication_title`
- `authors`, `editors`, `author_count`, `author_string`
- `tags`, `collections`, `zotero_uri`
- `attachments` (PDF-only)

**Omit-missing-keys rule:** optional fields that are absent from the Zotero item are omitted from the JSON entirely — they are never emitted as `null`.

**`tagging.include_abstract` flag:** when `true`, `abstract_note` is included in `paper_metadata`; when `false` (the default), it is omitted.

### Default Configuration

The full default tagging configuration (`zotero_docai_pipeline/conf/tagging/default.yaml`):

```yaml
# Tagging workflow configuration
selection:
  include:
    values:
      - docai
    operator: "or"
  exclude:
    values:
      - docai-processed
    operator: "or"
  conflict_resolution: "exclude_wins"

apply_on_success:
  values:
    - docai-processed

apply_on_error:
  values:
    - docai-error

include_abstract: false
```

### Overriding via CLI

Override tagging settings from the command line:

```bash
zotero-docai-pipeline "tagging.selection.include.values=[my-tag]"
# or: python -m zotero_docai_pipeline "tagging.selection.include.values=[my-tag]"
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
zotero-docai-pipeline tag_adding.enabled=true
# or: python -m zotero_docai_pipeline tag_adding.enabled=true
```

## PDF Download

Optional step that downloads PDFs from Zotero items to local disk (used as an input for OCR, and also supported in download-only mode).

### Key configuration
- `download.enabled`: set to `true` to enable PDF downloads.
- `download.upload_folder`: local directory for downloaded PDFs (must be overridden with an explicit path when `download.enabled=true`).
- `download.preserve_filenames`: whether to preserve the original PDF filenames (default: `true`).
- `download.create_subfolders`: whether to create subfolders under `upload_folder` (default: `false`).
- `download.skip_existing`: whether to skip PDFs that already exist locally (default: `true`).
- `download.max_concurrent_downloads`: maximum number of concurrent downloads (default: `5`).
- Retry is configurable via `download.retry.*` (see `zotero_docai_pipeline/conf/download/default.yaml`).

### Important constraint
- `processing.dry_run=true` cannot be combined with `download.enabled=true`.

Examples:
```bash
# Download-only (explicit output path required)
zotero-docai-pipeline download.enabled=true download.upload_folder=/path/to/downloads
# or: python -m zotero_docai_pipeline download.enabled=true download.upload_folder=/path/to/downloads

# Download + OCR
zotero-docai-pipeline download.enabled=true download.upload_folder=/path/to/downloads ocr.enabled=true
# or: python -m zotero_docai_pipeline download.enabled=true download.upload_folder=/path/to/downloads ocr.enabled=true
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

**API keys**: Ensure the correct environment variable is set (see [Runtime Prerequisites](#runtime-prerequisites)):
- `PAGEINDEX_API_KEY` for PageIndex OCR
- `MISTRAL_API_KEY` for Mistral OCR

**Tree extraction**: Requires PageIndex provider (`ocr.provider: pageindex`), valid `PAGEINDEX_API_KEY`, and `tree_structure.enabled: true` (see [Configuration](#configuration)).

### No Items Found

Verify:
- Items are tagged with the configured include tag(s) (see `tagging.selection.include.values`, default: `docai`)
- Items are not already tagged with the configured exclude tag(s) (see `tagging.selection.exclude.values`, default: `docai-processed`)
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
