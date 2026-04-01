"""Allow ``python -m zotero_docai_pipeline`` invocation."""

import sys

from zotero_docai_pipeline.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
